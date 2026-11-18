@echo off
echo ============================================
echo  Enable / refresh WSL2 NVIDIA GPU access
echo ============================================
echo.
echo [1/4] Checking Windows NVIDIA driver...
where nvidia-smi >nul 2>&1
if errorlevel 1 (
  echo ERROR: nvidia-smi not found on Windows PATH.
  echo Install/update the NVIDIA Game Ready or Studio driver, then rerun.
  pause
  exit /b 1
)
nvidia-smi
if errorlevel 1 (
  echo ERROR: Windows cannot talk to the GPU.
  echo Update NVIDIA driver from https://www.nvidia.com/Download/index.aspx
  pause
  exit /b 1
)
echo.
echo [2/4] Shutting down WSL (this closes all Linux terminals)...
wsl --shutdown
timeout /t 3 /nobreak >nul
echo.
echo [3/4] Starting WSL and testing GPU inside Linux...
wsl -e bash -lc "nvidia-smi; echo; ls -la /dev/dxg; echo; source /home/admin1/miniforge3/etc/profile.d/conda.sh && conda activate colpo && python -c \"import torch; print('cuda_available=', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')\""
echo.
echo [4/4] Done.
echo If cuda_available=True, double-click tools\start_cuda_retrain.bat to train on GPU.
pause
