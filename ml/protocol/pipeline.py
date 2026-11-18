"""Run the pre-specified patient-level protocol ladder."""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ml.protocol.clinical_models import (
    FEATURE_NAMES_STACK,
    apply_score_scaler,
    clinical_matrix,
    coefficient_table,
    fit_age_stats,
    fit_score_scaler,
    predict_logistic,
    select_logistic,
    stack_design_matrix,
)
from ml.protocol.cohort import Cohort, load_cohort, write_data_card
from ml.protocol.constants import (
    ATTN_HIDDEN,
    BATCH_SIZE,
    CENTER_SIGMA,
    CLINICAL_FEATURE_NAMES,
    CONFIRM_SPLIT_SEEDS,
    DROPOUT,
    FROZEN_NESTED_CANDIDATES,
    FUSION_C_GRID,
    LSE_TAUS,
    MIL_ATTN_HIDDEN,
    MIL_DROPOUT,
    MIL_EMBED_DIM,
    N_BOOTSTRAP,
    N_OUTER_FOLDS,
    PROTOCOL_VERSION,
    RUNS_DIR,
    SCREENING_SPLIT_SEEDS,
    STACK_IMAGE_CANDIDATES,
    experiment_dir,
)
from ml.protocol.encoders import build_protocol_model
from ml.protocol.gallery import generate_attention_gallery
from ml.protocol.io_utils import (
    load_json,
    save_csv,
    save_json,
    utc_now,
    write_environment_lock,
    write_text,
)
from ml.protocol.metrics import summarize_patient_predictions
from ml.protocol.mil import GatedAttentionMIL
from ml.protocol.objective import balanced_patient_sampler
from ml.protocol.pooling import persons_from_image_frame
from ml.protocol.postprocess import apply_fold_postprocess
from ml.protocol.report import write_experiment_report, write_global_report, write_model_card
from ml.protocol.specs import ExperimentSpec, LADDER, spec_by_id
from ml.protocol.splits import (
    assign_inner_val,
    filter_images,
    make_outer_splits,
    person_ids_for_role,
    write_split_manifest,
)
from ml.protocol.train_common import load_checkpoint, resolve_device
from ml.protocol.train_image import predict_image_frame, train_image_fold
from ml.protocol.train_mil import predict_bag_frame, train_mil_fold
from ml.protocol.train_ssl import train_ssl_fold


def _attach_counts(images: pd.DataFrame, persons: pd.DataFrame) -> pd.DataFrame:
    out = images.copy()
    counts = persons.set_index("person_id")["n_images"]
    out["n_images"] = out["person_id"].map(counts)
    return out


def _tag_oof(frame: pd.DataFrame, spec: ExperimentSpec, fold: int, seed: int, extra: dict | None = None) -> pd.DataFrame:
    out = frame.copy()
    out["experiment_id"] = spec.id
    out["model_name"] = spec.name
    out["outer_fold"] = int(fold)
    out["split_seed"] = int(seed)
    if extra:
        for k, v in extra.items():
            out[k] = v
    return out


def _write_status(root: Path, payload: dict[str, Any]) -> None:
    payload = dict(payload)
    payload["updated_utc"] = utc_now()
    save_json(root / "STATUS.json", payload)


class ProtocolRunner:
    def __init__(
        self,
        root: Path | None = None,
        device: str = "cuda",
        num_workers: int = 4,
        force: bool = False,
        smoke: bool = False,
        split_seeds: tuple[int, ...] | None = None,
    ) -> None:
        self.root = Path(root) if root else RUNS_DIR
        self.device = device
        self.num_workers = 0 if smoke else int(num_workers)
        self.force = force
        self.smoke = smoke
        self.split_seeds = split_seeds or SCREENING_SPLIT_SEEDS
        self.cohort: Cohort | None = None
        self.splits: dict[int, pd.DataFrame] = {}
        self.root.mkdir(parents=True, exist_ok=True)

    def setup(self) -> None:
        setup_dir = self.root / "setup"
        setup_dir.mkdir(parents=True, exist_ok=True)
        self.cohort = load_cohort()
        write_data_card(self.cohort, setup_dir / "DATA_CARD.md")
        write_environment_lock(setup_dir / "environment.lock")
        for seed in set(self.split_seeds) | set(CONFIRM_SPLIT_SEEDS):
            outer = make_outer_splits(self.cohort.persons, n_splits=N_OUTER_FOLDS, seed=seed)
            audit = write_split_manifest(outer, self.cohort.images, setup_dir, seed)
            self.splits[seed] = outer
            print(f"splits seed={seed} folds={audit['n_folds']} ok={audit['ok']}", flush=True)
        save_json(
            setup_dir / "protocol.json",
            {
                "version": PROTOCOL_VERSION,
                "n_outer_folds": N_OUTER_FOLDS,
                "split_seeds": list(self.split_seeds),
                "smoke": self.smoke,
            },
        )
        _write_status(self.root, {"phase": "setup", "ok": True})

    def _outer(self, seed: int) -> pd.DataFrame:
        if seed not in self.splits:
            assert self.cohort is not None
            self.splits[seed] = make_outer_splits(self.cohort.persons, seed=seed)
        return self.splits[seed]

    def _fold_frames(self, seed: int, fold: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        assert self.cohort is not None
        tagged = assign_inner_val(self._outer(seed), fold, seed=seed * 1009 + fold)
        persons = self.cohort.persons.merge(
            tagged[["person_id", "role", "outer_fold"]], on="person_id", how="left"
        )
        images = _attach_counts(self.cohort.images, self.cohort.persons)
        images = images.merge(tagged[["person_id", "role", "outer_fold"]], on="person_id", how="left")
        train = filter_images(images, person_ids_for_role(tagged, "inner_train"))
        val = filter_images(images, person_ids_for_role(tagged, "inner_val"))
        test = filter_images(images, person_ids_for_role(tagged, "outer_test"))
        return train, val, test, persons

    def _exp_done(self, spec: ExperimentSpec, seed: int) -> bool:
        path = experiment_dir(self.root, spec.id, seed) / "oof_predictions.csv"
        return path.is_file() and not self.force

    def _finalize_oof(self, spec: ExperimentSpec, seed: int, parts: list[pd.DataFrame], fold_summaries: list[dict]) -> pd.DataFrame:
        dest = experiment_dir(self.root, spec.id, seed)
        oof = pd.concat(parts, ignore_index=True)
        # Join legacy split / n_images
        assert self.cohort is not None
        meta = self.cohort.persons[["person_id", "legacy_split", "n_images", "age", "hpv_bucket", "cyto_bucket"]]
        keep = [c for c in oof.columns if c not in set(meta.columns) - {"person_id"}]
        oof = oof[keep].merge(meta, on="person_id", how="left")
        save_csv(dest / "oof_predictions.csv", oof)
        save_json(dest / "fold_summaries.json", fold_summaries)
        write_experiment_report(dest, spec.id, spec.name)
        write_global_report(self.root, n_boot=200 if self.smoke else N_BOOTSTRAP)
        return oof

    def run_clinical(self, spec: ExperimentSpec, seed: int) -> None:
        dest = experiment_dir(self.root, spec.id, seed)
        if self._exp_done(spec, seed):
            print(f"skip {spec.id} seed={seed} (exists)", flush=True)
            return
        parts, folds = [], []
        n_folds = 1 if self.smoke else N_OUTER_FOLDS
        for fold in range(n_folds):
            train, val, test, persons = self._fold_frames(seed, fold)
            tr_p = persons[persons["role"] == "inner_train"]
            va_p = persons[persons["role"] == "inner_val"]
            te_p = persons[persons["role"] == "outer_test"]
            age_mean, age_std = fit_age_stats(tr_p)
            x_tr = clinical_matrix(tr_p, age_mean, age_std)
            x_va = clinical_matrix(va_p, age_mean, age_std)
            x_te = clinical_matrix(te_p, age_mean, age_std)
            clf, best_c, info = select_logistic(
                x_tr, tr_p["person_label"].to_numpy(), x_va, va_p["person_label"].to_numpy()
            )
            z_va, p_va = predict_logistic(clf, x_va)
            z_te, p_te = predict_logistic(clf, x_te)
            val_df = va_p.rename(columns={"person_label": "label"}).copy()
            tes_df = te_p.rename(columns={"person_label": "label"}).copy()
            val_df["logit"], val_df["probability"] = z_va, p_va
            tes_df["logit"], tes_df["probability"] = z_te, p_te
            val_df, tes_df, post = apply_fold_postprocess(val_df, tes_df)
            tes_df = _tag_oof(tes_df, spec, fold, seed)
            tes_df["image_count"] = tes_df["n_images"]
            parts.append(tes_df)
            folds.append(
                {
                    "fold": fold,
                    "best_C": best_c,
                    "coefficients": coefficient_table(clf, CLINICAL_FEATURE_NAMES),
                    "postprocess": post,
                    **info,
                }
            )
            print(f"{spec.id} fold {fold} C={best_c} n_test={len(tes_df)}", flush=True)
        self._finalize_oof(spec, seed, parts, folds)

    def run_image(self, spec: ExperimentSpec, seed: int) -> None:
        dest = experiment_dir(self.root, spec.id, seed)
        if self._exp_done(spec, seed):
            print(f"skip {spec.id} seed={seed} (exists)", flush=True)
            return
        parts, folds = [], []
        n_folds = 1 if self.smoke else N_OUTER_FOLDS
        for fold in range(n_folds):
            train, val, test, _ = self._fold_frames(seed, fold)
            fold_dir = dest / f"fold_{fold}"
            sampler = balanced_patient_sampler(train) if spec.balanced_sampler else None
            print(f"\n=== {spec.id} {spec.name} seed={seed} fold={fold} ===", flush=True)
            result = train_image_fold(
                train_images=train,
                val_images=val,
                dataset_root=self.cohort.dataset_root,  # type: ignore[union-attr]
                out_dir=fold_dir,
                backbone=spec.backbone,
                model_type=spec.model_type,
                center_prior=spec.center_prior,
                loss_name=spec.loss,
                use_patient_weights=spec.patient_weights,
                sampler=sampler,
                pooling=spec.pooling,
                seed=seed + fold,
                device_name=self.device,
                num_workers=self.num_workers,
                smoke=self.smoke,
                batch_size=BATCH_SIZE if spec.backbone != "convnext_tiny" else 24,
            )
            model, device = result["model"], result["device"]
            test_img = predict_image_frame(
                model, test, self.cohort.dataset_root, device, self.num_workers  # type: ignore[union-attr]
            )
            val_img = result["val_predictions"]
            test_img.to_csv(fold_dir / "test_image_predictions.csv", index=False)
            val_p = persons_from_image_frame(val_img, reduce=spec.pooling)
            tes_p = persons_from_image_frame(test_img, reduce=spec.pooling)
            val_p, tes_p, post = apply_fold_postprocess(val_p, tes_p)
            tes_p = _tag_oof(tes_p, spec, fold, seed)
            parts.append(tes_p)
            folds.append({"fold": fold, "postprocess": post, "train_summary": result["summary"]})
            # keep image OOF for pooling / gallery
            test_img = _tag_oof(test_img, spec, fold, seed)
            test_img.to_csv(fold_dir / "test_image_oof.csv", index=False)
        self._finalize_oof(spec, seed, parts, folds)
        # concatenated image OOF
        img_parts = []
        for fold in range(n_folds):
            p = dest / f"fold_{fold}" / "test_image_oof.csv"
            if p.is_file():
                img_parts.append(pd.read_csv(p))
        if img_parts:
            save_csv(dest / "oof_image_predictions.csv", pd.concat(img_parts, ignore_index=True))

    def run_mil(self, spec: ExperimentSpec, seed: int, ssl_bodies: dict[int, dict] | None = None) -> None:
        dest = experiment_dir(self.root, spec.id, seed)
        if self._exp_done(spec, seed):
            print(f"skip {spec.id} seed={seed} (exists)", flush=True)
            return
        # Center prior for MIL: copy winner of B3 vs B4 if both exist.
        center = spec.center_prior
        if spec.id == "B6":
            center = self._best_center_prior(seed)
        parts, folds = [], []
        n_folds = 1 if self.smoke else N_OUTER_FOLDS
        for fold in range(n_folds):
            train, val, test, _ = self._fold_frames(seed, fold)
            fold_dir = dest / f"fold_{fold}"
            print(f"\n=== {spec.id} MIL seed={seed} fold={fold} center_prior={center} ===", flush=True)
            ssl_body = None if ssl_bodies is None else ssl_bodies.get(fold)
            result = train_mil_fold(
                train_images=train,
                val_images=val,
                dataset_root=self.cohort.dataset_root,  # type: ignore[union-attr]
                out_dir=fold_dir,
                backbone=spec.backbone,
                model_type=spec.model_type,
                center_prior=center,
                loss_name=spec.loss,
                seed=seed + fold,
                device_name=self.device,
                num_workers=self.num_workers,
                ssl_body=ssl_body,
                balanced_sampler=spec.balanced_sampler,
                smoke=self.smoke,
            )
            model, device = result["model"], result["device"]
            tes = predict_bag_frame(
                model, test, self.cohort.dataset_root, device, self.num_workers  # type: ignore[union-attr]
            )
            tes.to_csv(fold_dir / "test_person_predictions.csv", index=False)
            val_p = result["val_predictions"].copy()
            tes_p = tes.copy()
            if "person_label" in val_p.columns:
                val_p = val_p.drop(columns=["person_label"])
            if "person_label" in tes_p.columns:
                tes_p = tes_p.drop(columns=["person_label"])
            val_p, tes_p, post = apply_fold_postprocess(val_p, tes_p)
            tes_p = _tag_oof(tes_p, spec, fold, seed)
            parts.append(tes_p)
            folds.append({"fold": fold, "postprocess": post, "center_prior": center, "train_summary": result["summary"]})
        self._finalize_oof(spec, seed, parts, folds)

    def run_ssl_mil(self, spec: ExperimentSpec, seed: int) -> None:
        dest = experiment_dir(self.root, spec.id, seed)
        if self._exp_done(spec, seed):
            print(f"skip {spec.id} seed={seed} (exists)", flush=True)
            return
        bodies: dict[int, dict] = {}
        n_folds = 1 if self.smoke else N_OUTER_FOLDS
        for fold in range(n_folds):
            train, _, _, _ = self._fold_frames(seed, fold)
            ssl_dir = dest / f"fold_{fold}" / "ssl"
            print(f"\n=== {spec.id} SSL ({spec.ssl}) seed={seed} fold={fold} ===", flush=True)
            ssl = train_ssl_fold(
                train_images=train,
                dataset_root=self.cohort.dataset_root,  # type: ignore[union-attr]
                out_dir=ssl_dir,
                patient_aware=(spec.ssl == "micle"),
                seed=seed + fold,
                device_name=self.device,
                num_workers=self.num_workers,
                smoke=self.smoke,
            )
            bodies[fold] = ssl["body_state"]
        self.run_mil(spec, seed, ssl_bodies=bodies)

    def run_pooling(self, spec: ExperimentSpec, seed: int) -> None:
        dest = experiment_dir(self.root, spec.id, seed)
        if self._exp_done(spec, seed):
            print(f"skip {spec.id} seed={seed} (exists)", flush=True)
            return
        sources = ["B1", "B2", "B3", "B4"]
        all_parts: dict[str, list[pd.DataFrame]] = {}
        fold_summ: list[dict[str, Any]] = []
        n_folds = 1 if self.smoke else N_OUTER_FOLDS
        rules = [("max", None), ("mean", None)] + [(f"lse_{t}", t) for t in LSE_TAUS]
        for src in sources:
            src_dir = experiment_dir(self.root, src, seed)
            if not (src_dir / "oof_image_predictions.csv").is_file() and not any(
                (src_dir / f"fold_{f}" / "test_image_predictions.csv").is_file() for f in range(n_folds)
            ):
                continue
            for fold in range(n_folds):
                fold_dir = src_dir / f"fold_{fold}"
                val_img_p = fold_dir / "val_image_predictions.csv"
                tes_img_p = fold_dir / "test_image_predictions.csv"
                if not val_img_p.is_file() or not tes_img_p.is_file():
                    continue
                val_img = pd.read_csv(val_img_p)
                tes_img = pd.read_csv(tes_img_p)
                for name, tau in rules:
                    val_p = persons_from_image_frame(val_img, reduce=name if tau is None else "lse", tau=tau or 0.5)
                    tes_p = persons_from_image_frame(tes_img, reduce=name if tau is None else "lse", tau=tau or 0.5)
                    val_p, tes_p, post = apply_fold_postprocess(val_p, tes_p)
                    key = f"{src}_{name}"
                    tes_p = _tag_oof(
                        tes_p,
                        ExperimentSpec(id=f"B5_{key}", name=key, kind="pooling", description=""),
                        fold,
                        seed,
                    )
                    tes_p["source_experiment"] = src
                    tes_p["pooling"] = name
                    all_parts.setdefault(key, []).append(tes_p)
                    fold_summ.append({"source": src, "pooling": name, "fold": fold, "postprocess": post})
        if not all_parts:
            print("B5 skipped: no image OOF yet", flush=True)
            return
        dest.mkdir(parents=True, exist_ok=True)
        # Write one combined OOF plus per-rule files; pick best by AUROC for B5 headline.
        combined = []
        best_key, best_auc = None, -1.0
        from sklearn.metrics import roc_auc_score

        for key, parts in all_parts.items():
            df = pd.concat(parts, ignore_index=True)
            save_csv(dest / f"oof_{key}.csv", df)
            auc = float(roc_auc_score(df["label"], df["calibrated_probability"]))
            if auc > best_auc:
                best_auc, best_key = auc, key
            combined.append(df)
        headline = pd.concat(all_parts[best_key], ignore_index=True) if best_key else pd.concat(combined, ignore_index=True)
        headline["experiment_id"] = spec.id
        headline["model_name"] = f"pooling_best:{best_key}"
        save_json(dest / "pooling_choice.json", {"best": best_key, "best_auroc": best_auc})
        self._finalize_oof(spec, seed, [headline], fold_summ)

    def _best_center_prior(self, seed: int) -> bool:
        def auc_of(eid: str) -> float:
            p = experiment_dir(self.root, eid, seed) / "metrics.json"
            if not p.is_file():
                return float("nan")
            return float(load_json(p)["discrimination"]["auroc"])

        a3, a4 = auc_of("B3"), auc_of("B4")
        if a3 == a3 and a4 == a4:
            return bool(a4 >= a3)
        return True

    def _fold_image_person_tables(
        self, eid: str, seed: int, fold: int
    ) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
        """Saved inner-val / outer-test person scores for one image experiment fold."""
        src_spec = spec_by_id(eid)
        fold_dir = experiment_dir(self.root, eid, seed) / f"fold_{fold}"
        mil_val = fold_dir / "val_person_predictions.csv"
        mil_tes = fold_dir / "test_person_predictions.csv"
        val_src = fold_dir / "val_image_predictions.csv"
        tes_src = fold_dir / "test_image_predictions.csv"
        if mil_val.is_file() and mil_tes.is_file():
            val_p = pd.read_csv(mil_val)
            tes_p = pd.read_csv(mil_tes)
        elif val_src.is_file() and tes_src.is_file():
            pooling = src_spec.pooling
            val_p = persons_from_image_frame(pd.read_csv(val_src), reduce=pooling)
            tes_p = persons_from_image_frame(pd.read_csv(tes_src), reduce=pooling)
        else:
            return None, None
        for frame in (val_p, tes_p):
            if "label" not in frame.columns and "person_label" in frame.columns:
                frame["label"] = frame["person_label"]
        return val_p, tes_p

    def _inner_val_auroc(self, eid: str, seed: int, fold: int) -> float:
        val_p, _ = self._fold_image_person_tables(eid, seed, fold)
        if val_p is None or "label" not in val_p.columns or "logit" not in val_p.columns:
            return float("nan")
        y = val_p["label"].to_numpy()
        if np.unique(y).size < 2:
            return float("nan")
        return float(roc_auc_score(y, val_p["logit"].to_numpy()))

    def _best_image_spec_fold(self, seed: int, fold: int) -> tuple[str, float]:
        """Pick image source using this fold's inner-val only (not full OOF)."""
        best, best_auc = "B4", float("nan")
        for eid in STACK_IMAGE_CANDIDATES:
            auc = self._inner_val_auroc(eid, seed, fold)
            if np.isfinite(auc) and (not np.isfinite(best_auc) or auc > best_auc):
                best, best_auc = eid, auc
        return best, best_auc

    def _fold_center_prior(self, eid: str, seed: int, fold: int) -> bool:
        spec = spec_by_id(eid)
        path = experiment_dir(self.root, eid, seed) / "fold_summaries.json"
        if path.is_file():
            for row in load_json(path):
                if int(row.get("fold", -1)) == int(fold) and "center_prior" in row:
                    return bool(row["center_prior"])
        return bool(spec.center_prior)

    def _infer_inner_train_scores(
        self, eid: str, seed: int, fold: int, train_images: pd.DataFrame
    ) -> pd.DataFrame | None:
        """Score inner-train people with the already-trained fold checkpoint."""
        spec = spec_by_id(eid)
        ckpt_path = experiment_dir(self.root, eid, seed) / f"fold_{fold}" / "best.pt"
        if not ckpt_path.is_file() or self.cohort is None:
            return None
        device = resolve_device(self.device)
        center = self._fold_center_prior(eid, seed, fold)
        ckpt = load_checkpoint(ckpt_path, device)
        if spec.kind == "mil":
            encoder = build_protocol_model(
                backbone=spec.backbone,
                model_type=spec.model_type,
                pretrained=False,
                dropout=DROPOUT,
                attn_hidden=ATTN_HIDDEN,
                center_prior=center,
                center_sigma=CENTER_SIGMA,
            )
            model = GatedAttentionMIL(
                encoder,
                embed_dim=MIL_EMBED_DIM,
                attn_hidden=MIL_ATTN_HIDDEN,
                dropout=MIL_DROPOUT,
            )
            model.load_state_dict(ckpt["model_state"])
            model.to(device).eval()
            return predict_bag_frame(
                model, train_images, self.cohort.dataset_root, device, self.num_workers
            )
        model = build_protocol_model(
            backbone=spec.backbone,
            model_type=spec.model_type,
            pretrained=False,
            dropout=DROPOUT,
            attn_hidden=ATTN_HIDDEN,
            center_prior=center,
            center_sigma=CENTER_SIGMA,
        )
        model.load_state_dict(ckpt["model_state"])
        model.to(device).eval()
        img = predict_image_frame(
            model, train_images, self.cohort.dataset_root, device, self.num_workers
        )
        return persons_from_image_frame(img, reduce=spec.pooling)

    def _attach_clinical(
        self, scores: pd.DataFrame, people: pd.DataFrame
    ) -> pd.DataFrame:
        clin_cols = ["age", "hpv_bucket", "cyto_bucket"]
        out = scores.drop(columns=[c for c in clin_cols if c in scores.columns], errors="ignore")
        if "person_label" in out.columns:
            out = out.drop(columns=["person_label"])
        keep = ["person_id", *clin_cols]
        out = out.merge(people[keep], on="person_id", how="left")
        if "label" not in out.columns:
            lab = people[["person_id", "person_label"]].rename(columns={"person_label": "label"})
            out = out.merge(lab, on="person_id", how="left")
        return out

    def run_stack(self, spec: ExperimentSpec, seed: int) -> None:
        dest = experiment_dir(self.root, spec.id, seed)
        if self._exp_done(spec, seed):
            print(f"skip {spec.id} seed={seed} (exists)", flush=True)
            return
        parts, folds = [], []
        n_folds = 1 if self.smoke else N_OUTER_FOLDS
        for fold in range(n_folds):
            train, val, test, persons = self._fold_frames(seed, fold)
            tr_p = persons[persons["role"] == "inner_train"]
            va_p = persons[persons["role"] == "inner_val"]
            te_p = persons[persons["role"] == "outer_test"]
            age_mean, age_std = fit_age_stats(tr_p)

            # Clinical-only nested baseline (same as B0, this fold).
            x_tr_c = clinical_matrix(tr_p, age_mean, age_std)
            x_va_c = clinical_matrix(va_p, age_mean, age_std)
            x_te_c = clinical_matrix(te_p, age_mean, age_std)
            clin_clf, clin_c, clin_info = select_logistic(
                x_tr_c,
                tr_p["person_label"].to_numpy(),
                x_va_c,
                va_p["person_label"].to_numpy(),
            )
            _, p_va_c = predict_logistic(clin_clf, x_va_c)
            clin_val_auc = (
                float(roc_auc_score(va_p["person_label"], p_va_c))
                if np.unique(va_p["person_label"]).size > 1
                else float("nan")
            )

            src, img_val_auc = self._best_image_spec_fold(seed, fold)
            print(
                f"B11 fold {fold}: source={src} inner-val image AUROC={img_val_auc:.3f} "
                f"clinical={clin_val_auc:.3f}",
                flush=True,
            )
            val_p, tes_p = self._fold_image_person_tables(src, seed, fold)
            if val_p is None or tes_p is None:
                print(f"stack fold {fold}: missing source preds", flush=True)
                continue
            val_p = self._attach_clinical(val_p, va_p)
            tes_p = self._attach_clinical(tes_p, te_p)
            tr_img = self._infer_inner_train_scores(src, seed, fold, train)
            used_train = tr_img is not None and len(tr_img) >= 8
            if used_train:
                tr_p_img = self._attach_clinical(tr_img, tr_p)
                score_mean, score_std = fit_score_scaler(tr_p_img["logit"].to_numpy())
                tr_p_img = tr_p_img.copy()
                val_p = val_p.copy()
                tes_p = tes_p.copy()
                tr_p_img["image_score"] = apply_score_scaler(tr_p_img["logit"].to_numpy(), score_mean, score_std)
                val_p["image_score"] = apply_score_scaler(val_p["logit"].to_numpy(), score_mean, score_std)
                tes_p["image_score"] = apply_score_scaler(tes_p["logit"].to_numpy(), score_mean, score_std)
                x_tr = stack_design_matrix(tr_p_img, age_mean, age_std, image_score_col="image_score")
                x_va = stack_design_matrix(val_p, age_mean, age_std, image_score_col="image_score")
                x_te = stack_design_matrix(tes_p, age_mean, age_std, image_score_col="image_score")
                y_tr = tr_p_img["label"].to_numpy()
                clf, best_c, info = select_logistic(
                    x_tr,
                    y_tr,
                    x_va,
                    val_p["label"].to_numpy(),
                    c_grid=FUSION_C_GRID,
                    refit_with_val=True,
                )
            else:
                print(f"stack fold {fold}: no inner-train scores; val-only fit", flush=True)
                score_mean, score_std = fit_score_scaler(val_p["logit"].to_numpy())
                val_p = val_p.copy()
                tes_p = tes_p.copy()
                val_p["image_score"] = apply_score_scaler(val_p["logit"].to_numpy(), score_mean, score_std)
                tes_p["image_score"] = apply_score_scaler(tes_p["logit"].to_numpy(), score_mean, score_std)
                x_va = stack_design_matrix(val_p, age_mean, age_std, image_score_col="image_score")
                x_te = stack_design_matrix(tes_p, age_mean, age_std, image_score_col="image_score")
                clf, best_c, info = select_logistic(
                    x_va,
                    val_p["label"].to_numpy(),
                    x_va,
                    val_p["label"].to_numpy(),
                    c_grid=FUSION_C_GRID,
                    refit_with_val=False,
                )

            z_va, p_va = predict_logistic(clf, x_va)
            stack_val_auc = (
                float(roc_auc_score(val_p["label"], p_va)) if np.unique(val_p["label"]).size > 1 else float("nan")
            )
            use_clinical = np.isfinite(clin_val_auc) and (
                not np.isfinite(stack_val_auc) or stack_val_auc < clin_val_auc
            )
            if use_clinical:
                print(
                    f"B11 fold {fold}: stack val AUROC {stack_val_auc:.3f} < clinical "
                    f"{clin_val_auc:.3f} → emit clinical-only",
                    flush=True,
                )
                z_va, p_va = predict_logistic(clin_clf, x_va_c)
                z_te, p_te = predict_logistic(clin_clf, x_te_c)
                val_df = va_p.rename(columns={"person_label": "label"}).copy()
                tes_df = te_p.rename(columns={"person_label": "label"}).copy()
                chosen = "clinical"
                emit_c = clin_c
                coefs = coefficient_table(clin_clf, CLINICAL_FEATURE_NAMES)
                info = clin_info
            else:
                z_te, p_te = predict_logistic(clf, x_te)
                val_df = val_p.copy()
                tes_df = tes_p.copy()
                chosen = src
                emit_c = best_c
                coefs = coefficient_table(clf, FEATURE_NAMES_STACK)

            val_df["logit"], val_df["probability"] = z_va, p_va
            tes_df["logit"], tes_df["probability"] = z_te, p_te
            val_df, tes_df, post = apply_fold_postprocess(val_df, tes_df)
            tes_df = _tag_oof(
                tes_df,
                spec,
                fold,
                seed,
                extra={"stack_source": chosen, "image_candidate": src},
            )
            parts.append(tes_df)
            folds.append(
                {
                    "fold": fold,
                    "source": chosen,
                    "image_candidate": src,
                    "image_val_auroc": img_val_auc,
                    "clinical_val_auroc": clin_val_auc,
                    "stack_val_auroc": stack_val_auc,
                    "used_inner_train_scores": used_train,
                    "best_C": emit_c,
                    "coefficients": coefs,
                    "postprocess": post,
                    **info,
                }
            )
        if parts:
            self._finalize_oof(spec, seed, parts, folds)

    def run_gallery(self, spec_id: str = "B4", seed: int = 42) -> None:
        dest = experiment_dir(self.root, spec_id, seed)
        oof_img = dest / "oof_image_predictions.csv"
        oof_person = dest / "oof_predictions.csv"
        if not oof_img.is_file() or not oof_person.is_file():
            print("gallery skipped: missing OOF", flush=True)
            return
        spec = spec_by_id(spec_id)
        imgs = pd.read_csv(oof_img)
        pers = pd.read_csv(oof_person)[["person_id", "calibrated_probability", "predicted_class_op70", "label", "outer_fold"]]
        pers = pers.rename(columns={"label": "person_label_oof"})
        merged = imgs.merge(pers, on=["person_id", "outer_fold"], how="left")
        merged["person_label"] = merged["person_label"]
        merged["pred70"] = merged["predicted_class_op70"]
        ckpts = []
        n_folds = 1 if self.smoke else N_OUTER_FOLDS
        for fold in range(n_folds):
            p = dest / f"fold_{fold}" / "best.pt"
            if p.is_file():
                ckpts.append(
                    {
                        "fold": fold,
                        "path": str(p),
                        "backbone": spec.backbone,
                        "model_type": spec.model_type,
                        "center_prior": spec.center_prior,
                    }
                )
        generate_attention_gallery(
            fold_ckpts=ckpts,
            oof=merged,
            dataset_root=self.cohort.dataset_root,  # type: ignore[union-attr]
            out_dir=self.root / "galleries" / f"{spec_id}_seed{seed}",
            device_name=self.device,
        )

    def _nested_val_auroc(self, eid: str, seed: int, fold: int) -> float:
        """Inner-val AUROC for nested selection. Outer test is never used."""
        if eid == "B0":
            path = experiment_dir(self.root, "B0", seed) / "fold_summaries.json"
            if path.is_file():
                for row in load_json(path):
                    if int(row.get("fold", -1)) == int(fold) and "best_val_auroc" in row:
                        auc = float(row["best_val_auroc"])
                        if np.isfinite(auc):
                            return auc
            train, _, _, persons = self._fold_frames(seed, fold)
            tr_p = persons[persons["role"] == "inner_train"]
            va_p = persons[persons["role"] == "inner_val"]
            age_mean, age_std = fit_age_stats(tr_p)
            _, _, info = select_logistic(
                clinical_matrix(tr_p, age_mean, age_std),
                tr_p["person_label"].to_numpy(),
                clinical_matrix(va_p, age_mean, age_std),
                va_p["person_label"].to_numpy(),
                refit_with_val=False,
            )
            return float(info.get("best_val_auroc", float("nan")))
        if eid == "B11":
            path = experiment_dir(self.root, "B11", seed) / "fold_summaries.json"
            if path.is_file():
                for row in load_json(path):
                    if int(row.get("fold", -1)) != int(fold):
                        continue
                    # C-search val AUROC (train-fit), comparable to B0. Do not use
                    # stack_val_auroc after refit-on-val.
                    for key in ("best_val_auroc", "stack_val_auroc", "clinical_val_auroc"):
                        if row.get("source") == "clinical" and key == "stack_val_auroc":
                            continue
                        val = row.get(key)
                        if val is not None and np.isfinite(float(val)):
                            return float(val)
            return float("nan")
        return self._inner_val_auroc(eid, seed, fold)

    def run_nested(self, spec: ExperimentSpec, seed: int) -> None:
        """Select among frozen candidates on inner-val; evaluate on outer test."""
        dest = experiment_dir(self.root, spec.id, seed)
        if self._exp_done(spec, seed):
            print(f"skip {spec.id} seed={seed} (exists)", flush=True)
            return
        available = []
        for eid in FROZEN_NESTED_CANDIDATES:
            oof_p = experiment_dir(self.root, eid, seed) / "oof_predictions.csv"
            if oof_p.is_file():
                available.append(eid)
        if not available:
            print(f"{spec.id} seed={seed}: no frozen OOF tables", flush=True)
            return
        print(f"B12 nested candidates={available}", flush=True)
        oofs = {eid: pd.read_csv(experiment_dir(self.root, eid, seed) / "oof_predictions.csv") for eid in available}
        parts, folds = [], []
        n_folds = 1 if self.smoke else N_OUTER_FOLDS
        for fold in range(n_folds):
            scores = {eid: self._nested_val_auroc(eid, seed, fold) for eid in available}
            # Simplicity tie-break: B0, then B1, then B4, then B11.
            ranked = sorted(
                available,
                key=lambda e: (
                    -scores[e] if np.isfinite(scores[e]) else 1e9,
                    FROZEN_NESTED_CANDIDATES.index(e) if e in FROZEN_NESTED_CANDIDATES else 99,
                ),
            )
            winner = ranked[0]
            src = oofs[winner]
            if "outer_fold" not in src.columns:
                print(f"B12 fold {fold}: {winner} missing outer_fold", flush=True)
                continue
            tes = src[src["outer_fold"].astype(int) == int(fold)].copy()
            if tes.empty:
                print(f"B12 fold {fold}: no test rows from {winner}", flush=True)
                continue
            tes = tes.drop(columns=[c for c in tes.columns if c in {"experiment_id", "model_name", "stack_source", "image_candidate"}], errors="ignore")
            tes = _tag_oof(tes, spec, fold, seed, extra={"nested_choice": winner})
            parts.append(tes)
            folds.append(
                {
                    "fold": fold,
                    "choice": winner,
                    "inner_val_auroc": {e: float(scores[e]) if np.isfinite(scores[e]) else None for e in available},
                }
            )
            pretty = " ".join(f"{e}={scores[e]:.3f}" if np.isfinite(scores[e]) else f"{e}=nan" for e in available)
            print(f"B12 fold {fold}: pick {winner}  [{pretty}]  n_test={len(tes)}", flush=True)
        if parts:
            self._finalize_oof(spec, seed, parts, folds)

    def run_spec(self, spec: ExperimentSpec, seed: int) -> None:
        _write_status(self.root, {"phase": spec.id, "seed": seed, "name": spec.name})
        print(f"\n######## {spec.id} {spec.name} seed={seed} ########", flush=True)
        if spec.kind == "clinical":
            self.run_clinical(spec, seed)
        elif spec.kind == "image":
            self.run_image(spec, seed)
        elif spec.kind == "mil":
            self.run_mil(spec, seed)
        elif spec.kind == "ssl_mil":
            self.run_ssl_mil(spec, seed)
        elif spec.kind == "pooling":
            self.run_pooling(spec, seed)
        elif spec.kind == "stack":
            self.run_stack(spec, seed)
        elif spec.kind == "nested":
            self.run_nested(spec, seed)
        else:
            raise ValueError(spec.kind)

    def run_all(self, ids: list[str] | None = None) -> None:
        self.setup()
        specs = [s for s in LADDER if ids is None or s.id in ids]
        for seed in self.split_seeds:
            for spec in specs:
                try:
                    self.run_spec(spec, seed)
                except Exception:
                    tb = traceback.format_exc()
                    print(tb, flush=True)
                    log = self.root / "LOGS" / f"{spec.id}_seed{seed}.error.txt"
                    write_text(log, tb)
                    _write_status(self.root, {"phase": spec.id, "seed": seed, "error": True})
                    if self.smoke:
                        raise
        # Confirmation seeds for marked finalists after screening seed 42.
        if not self.smoke and ids is None:
            for spec in specs:
                if not spec.confirm:
                    continue
                for seed in CONFIRM_SPLIT_SEEDS:
                    if seed in self.split_seeds:
                        continue
                    try:
                        self.run_spec(spec, seed)
                    except Exception:
                        tb = traceback.format_exc()
                        write_text(self.root / "LOGS" / f"{spec.id}_seed{seed}.error.txt", tb)
        # Avoid regenerating attention galleries on subset runs (e.g. nested B12).
        if ids is None or any(x in ids for x in ("B3", "B4")):
            try:
                self.run_gallery("B4", list(self.split_seeds)[0])
                if (experiment_dir(self.root, "B3", list(self.split_seeds)[0]) / "oof_predictions.csv").is_file():
                    self.run_gallery("B3", list(self.split_seeds)[0])
            except Exception:
                traceback.print_exc()
        write_global_report(self.root, n_boot=200 if self.smoke else N_BOOTSTRAP)
        # Choose simplest among near-ties later; for now highest seed42 AUROC.
        board = self.root / "comparisons" / "leaderboard.csv"
        chosen = "undetermined"
        if board.is_file():
            tab = pd.read_csv(board)
            tab42 = tab[tab["seed"] == "seed42"] if "seed" in tab.columns else tab
            tab42 = tab42[~tab42["experiment_id"].astype(str).isin({"B12"})]
            if len(tab42):
                chosen = str(tab42.sort_values("auroc", ascending=False).iloc[0]["experiment_id"])
        write_model_card(self.root, chosen)
        _write_status(self.root, {"phase": "done", "chosen": chosen, "ok": True})
        print(f"\nProtocol complete. Results: {self.root / 'index.html'}", flush=True)
