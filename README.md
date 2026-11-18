# Colposcopy CIN2+ classification

Training and evaluation code for patient-level CIN2+ classification from colposcopy images.

Clinical images, patient identifiers, manifests, and experiment archives are not included.

## Layout

- `ml/models` model definitions
- `ml/protocol` training and evaluation pipeline
- `ml/configs` experiment configs
- `ml/scripts` training and evaluation scripts
- `ml/utils` shared helpers
- `cascade_system` notes for the cascade pipeline
- `tools` local GPU launch scripts that do not contain data paths

## Run

Install PyTorch, then run a training entry point from `ml/scripts` or `ml/protocol` with your own dataset path. Do not commit images or patient tables.
