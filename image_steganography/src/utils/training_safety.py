"""
Pre-Training Safety Checker for GAN Steganography
Verifies checkpoint integrity, GPU safety, and training conditions before starting.
"""
import torch
import psutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Optional
import json


class TrainingSafetyChecker:
    """Comprehensive safety checks before training starts."""
    
    def __init__(self, checkpoint_dir: str = "checkpoints/gan"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checks_passed = []
        self.checks_failed = []
        self.warnings = []
        
    def check_all(self, resume_checkpoint: Optional[str] = None) -> bool:
        """
        Run all safety checks.
        
        Returns:
            True if all critical checks pass, False otherwise
        """
        print("=" * 70)
        print("PRE-TRAINING SAFETY CHECKLIST")
        print("=" * 70)
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Run all checks
        self._check_checkpoint_directory()
        self._check_gpu_availability()
        self._check_gpu_memory()
        
        if resume_checkpoint:
            self._check_resume_checkpoint(resume_checkpoint)
        
        self._check_system_resources()
        self._check_laptop_safety()
        
        # Print results
        self._print_results()
        
        # Return overall status
        return len(self.checks_failed) == 0
    
    def _check_checkpoint_directory(self):
        """Verify checkpoint directory exists and is writable."""
        check_name = "Checkpoint Directory"
        
        if not self.checkpoint_dir.exists():
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            self.warnings.append(f"{check_name}: Created missing directory")
        
        # Check write permissions
        test_file = self.checkpoint_dir / ".write_test"
        try:
            test_file.touch()
            test_file.unlink()
            self.checks_passed.append(f"{check_name}: ✅ Writable")
        except Exception as e:
            self.checks_failed.append(f"{check_name}: ❌ Not writable - {e}")
    
    def _check_gpu_availability(self):
        """Check CUDA availability."""
        check_name = "GPU Availability"
        
        if not torch.cuda.is_available():
            self.checks_failed.append(f"{check_name}: ❌ CUDA not available")
            return
        
        gpu_name = torch.cuda.get_device_name(0)
        self.checks_passed.append(f"{check_name}: ✅ {gpu_name}")
    
    def _check_gpu_memory(self):
        """Check GPU memory usage and availability."""
        check_name = "GPU Memory"
        
        if not torch.cuda.is_available():
            return
        
        props = torch.cuda.get_device_properties(0)
        total_vram = props.total_memory / 1e9
        allocated = torch.cuda.memory_allocated(0) / 1e9
        reserved = torch.cuda.memory_reserved(0) / 1e9
        
        usage_pct = (reserved / total_vram) * 100
        
        if usage_pct > 80:
            self.checks_failed.append(
                f"{check_name}: ❌ High usage ({usage_pct:.1f}%) - Risk of OOM"
            )
        elif usage_pct > 60:
            self.warnings.append(
                f"{check_name}: ⚠️ Moderate usage ({usage_pct:.1f}%)"
            )
        else:
            self.checks_passed.append(
                f"{check_name}: ✅ {total_vram:.1f} GB total, {usage_pct:.1f}% used"
            )
    
    def _check_resume_checkpoint(self, checkpoint_path: str):
        """Verify resume checkpoint integrity."""
        check_name = "Resume Checkpoint"
        
        cp_path = Path(checkpoint_path)
        if not cp_path.exists():
            self.checks_failed.append(f"{check_name}: ❌ File not found: {checkpoint_path}")
            return
        
        try:
            cp = torch.load(cp_path, map_location='cpu', weights_only=False)
            
            # Check required keys
            required_keys = [
                'encoder_state_dict',
                'decoder_state_dict',
                'discriminator_state_dict',
                'optimizer_g_state_dict',
                'optimizer_d_state_dict',
                'epoch'
            ]
            
            missing_keys = [k for k in required_keys if k not in cp]
            
            if missing_keys:
                self.checks_failed.append(
                    f"{check_name}: ❌ Missing keys: {missing_keys}"
                )
                return
            
            # Check AMP scaler
            if 'scaler_state_dict' not in cp:
                self.warnings.append(
                    f"{check_name}: ⚠️ No AMP scaler state (may cause instability)"
                )
            
            epoch = cp.get('epoch', '?')
            losses = cp.get('losses', {})
            
            self.checks_passed.append(
                f"{check_name}: ✅ Epoch {epoch}, G Loss: {losses.get('g_loss', 'N/A'):.4f}"
            )
            
        except Exception as e:
            self.checks_failed.append(f"{check_name}: ❌ Load failed - {e}")
    
    def _check_system_resources(self):
        """Check system RAM and CPU."""
        check_name = "System Resources"
        
        # RAM check
        ram = psutil.virtual_memory()
        ram_available_gb = ram.available / 1e9
        ram_usage_pct = ram.percent
        
        if ram_usage_pct > 90:
            self.warnings.append(
                f"{check_name}: ⚠️ High RAM usage ({ram_usage_pct:.1f}%)"
            )
        
        # CPU check
        cpu_pct = psutil.cpu_percent(interval=1)
        
        self.checks_passed.append(
            f"{check_name}: ✅ RAM: {ram_available_gb:.1f} GB free, CPU: {cpu_pct:.1f}%"
        )
    
    def _check_laptop_safety(self):
        """Laptop-specific safety checks."""
        check_name = "Laptop Safety"
        
        # Check battery status
        battery = psutil.sensors_battery()
        
        if battery is None:
            self.checks_passed.append(f"{check_name}: ✅ Desktop system (no battery)")
            return
        
        if not battery.power_plugged:
            self.checks_failed.append(
                f"{check_name}: ❌ NOT PLUGGED IN - Training will drain battery!"
            )
        else:
            self.checks_passed.append(
                f"{check_name}: ✅ AC power connected ({battery.percent:.0f}% charged)"
            )
        
        # Warning about overnight training
        self.warnings.append(
            f"{check_name}: ⚠️ Ensure sleep/hibernation is DISABLED for overnight training"
        )
    
    def _print_results(self):
        """Print formatted results."""
        print("\n" + "-" * 70)
        print("RESULTS")
        print("-" * 70)
        
        if self.checks_passed:
            print("\n✅ PASSED CHECKS:")
            for check in self.checks_passed:
                print(f"  {check}")
        
        if self.warnings:
            print("\n⚠️ WARNINGS:")
            for warning in self.warnings:
                print(f"  {warning}")
        
        if self.checks_failed:
            print("\n❌ FAILED CHECKS:")
            for failure in self.checks_failed:
                print(f"  {failure}")
        
        print("\n" + "=" * 70)
        
        if self.checks_failed:
            print("STATUS: ❌ ABORT - Critical checks failed")
            print("=" * 70)
        else:
            print("STATUS: ✅ SAFE TO PROCEED")
            if self.warnings:
                print("Note: Review warnings above")
            print("=" * 70)


def verify_checkpoint_policy(checkpoint_dir: str = "checkpoints/gan") -> Dict:
    """
    Verify checkpoint saving policy is correct.
    
    Returns:
        Dict with checkpoint info
    """
    cp_dir = Path(checkpoint_dir)
    
    # Find all checkpoints
    epoch_checkpoints = sorted(cp_dir.glob("gan_checkpoint_epoch_*.pth"))
    best_checkpoint = cp_dir / "best_gan_model.pth"
    interrupted_checkpoint = cp_dir / "interrupted_checkpoint.pth"
    
    info = {
        'total_epoch_checkpoints': len(epoch_checkpoints),
        'has_best_checkpoint': best_checkpoint.exists(),
        'has_interrupted_checkpoint': interrupted_checkpoint.exists(),
        'latest_epoch': None,
        'checkpoint_count_ok': len(epoch_checkpoints) >= 3
    }
    
    if epoch_checkpoints:
        # Extract epoch number from filename
        latest = epoch_checkpoints[-1]
        epoch_num = int(latest.stem.split('_')[-1])
        info['latest_epoch'] = epoch_num
    
    print("\n" + "-" * 70)
    print("CHECKPOINT POLICY VERIFICATION")
    print("-" * 70)
    print(f"Epoch checkpoints: {info['total_epoch_checkpoints']}")
    print(f"Best checkpoint: {'✅ Exists' if info['has_best_checkpoint'] else '❌ Missing'}")
    print(f"Interrupted checkpoint: {'✅ Exists' if info['has_interrupted_checkpoint'] else '⚠️ None'}")
    print(f"Latest epoch: {info['latest_epoch']}")
    print(f"Recovery points: {'✅ Sufficient (≥3)' if info['checkpoint_count_ok'] else '⚠️ Limited (<3)'}")
    print("-" * 70)
    
    return info


if __name__ == '__main__':
    import sys
    
    # Get resume checkpoint from command line if provided
    resume_cp = sys.argv[1] if len(sys.argv) > 1 else None
    
    # Run safety checks
    checker = TrainingSafetyChecker()
    safe = checker.check_all(resume_checkpoint=resume_cp)
    
    # Verify checkpoint policy
    verify_checkpoint_policy()
    
    # Exit with appropriate code
    sys.exit(0 if safe else 1)
