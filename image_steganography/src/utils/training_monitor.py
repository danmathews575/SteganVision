"""
Training Monitor for GAN Steganography
Monitors loss stability, detects anomalies, and enforces abort conditions.
"""
import torch
import numpy as np
from collections import deque
from typing import Dict, List, Optional
from datetime import datetime


class TrainingMonitor:
    """Monitor training progress and detect anomalies."""
    
    def __init__(
        self,
        warmup_batches: int = 500,
        loss_history_size: int = 10,
        abort_on_nan: bool = True
    ):
        """
        Args:
            warmup_batches: Number of batches to ignore for stability checks after resume
            loss_history_size: Number of epochs to track for trend analysis
            abort_on_nan: Whether to abort on NaN/Inf detection
        """
        self.warmup_batches = warmup_batches
        self.loss_history_size = loss_history_size
        self.abort_on_nan = abort_on_nan
        
        # Loss history (epoch-level)
        self.g_loss_history = deque(maxlen=loss_history_size)
        self.d_loss_history = deque(maxlen=loss_history_size)
        self.cover_loss_history = deque(maxlen=loss_history_size)
        self.secret_loss_history = deque(maxlen=loss_history_size)
        
        # Batch counter (for warmup)
        self.batch_count = 0
        self.in_warmup = True
        
        # Anomaly tracking
        self.anomalies = []
        self.abort_triggered = False
        self.abort_reason = None
    
    def update(
        self,
        epoch: int,
        g_loss: float,
        d_loss: float,
        cover_loss: float,
        secret_loss: float,
        batch_idx: Optional[int] = None
    ) -> Dict:
        """
        Update monitor with new loss values.
        
        Returns:
            Dict with status and recommendations
        """
        # Check for NaN/Inf
        if self._check_nan_inf(g_loss, d_loss, cover_loss, secret_loss):
            return {
                'status': 'ABORT',
                'reason': 'NaN or Inf detected in losses',
                'should_stop': True
            }
        
        # Update batch count for warmup
        if batch_idx is not None:
            self.batch_count += 1
            if self.batch_count > self.warmup_batches:
                self.in_warmup = False
        
        # Store epoch-level losses
        self.g_loss_history.append(g_loss)
        self.d_loss_history.append(d_loss)
        self.cover_loss_history.append(cover_loss)
        self.secret_loss_history.append(secret_loss)
        
        # Skip stability checks during warmup
        if self.in_warmup:
            return {
                'status': 'WARMUP',
                'warmup_remaining': max(0, self.warmup_batches - self.batch_count),
                'should_stop': False
            }
        
        # Check for training instability
        instability = self._check_instability()
        if instability:
            self.anomalies.append({
                'epoch': epoch,
                'type': 'instability',
                'details': instability
            })
            
            # Abort if cover loss is increasing (critical)
            if 'cover_loss_increasing' in instability:
                self.abort_triggered = True
                self.abort_reason = "Cover loss increasing - imperceptibility degrading"
                return {
                    'status': 'ABORT',
                    'reason': self.abort_reason,
                    'should_stop': True
                }
        
        # Check for convergence
        converged = self._check_convergence()
        
        return {
            'status': 'STABLE' if not instability else 'UNSTABLE',
            'converged': converged,
            'should_stop': False,
            'anomalies': self.anomalies[-5:]  # Last 5 anomalies
        }
    
    def _check_nan_inf(self, *losses) -> bool:
        """Check if any loss is NaN or Inf."""
        for loss in losses:
            if np.isnan(loss) or np.isinf(loss):
                if self.abort_on_nan:
                    self.abort_triggered = True
                    self.abort_reason = f"NaN/Inf detected: {loss}"
                return True
        return False
    
    def _check_instability(self) -> Optional[str]:
        """
        Check for training instability.
        
        Returns:
            Description of instability if detected, None otherwise
        """
        if len(self.cover_loss_history) < 3:
            return None
        
        instabilities = []
        
        # Check if cover loss is increasing (BAD - imperceptibility degrading)
        recent_cover = list(self.cover_loss_history)[-3:]
        if all(recent_cover[i] < recent_cover[i+1] for i in range(len(recent_cover)-1)):
            instabilities.append("cover_loss_increasing")
        
        # Check for wild oscillations in G loss
        if len(self.g_loss_history) >= 5:
            g_losses = list(self.g_loss_history)[-5:]
            g_std = np.std(g_losses)
            g_mean = np.mean(g_losses)
            if g_std > 0.5 * g_mean:  # High variance
                instabilities.append("generator_oscillating")
        
        # Check for discriminator collapse
        if len(self.d_loss_history) >= 3:
            recent_d = list(self.d_loss_history)[-3:]
            if all(d < 0.01 for d in recent_d):
                instabilities.append("discriminator_collapsed")
        
        return ", ".join(instabilities) if instabilities else None
    
    def _check_convergence(self) -> bool:
        """
        Check if training has converged.
        
        Returns:
            True if losses are stable (converged)
        """
        if len(self.g_loss_history) < self.loss_history_size:
            return False
        
        # Check if losses are stable (low variance)
        g_std = np.std(list(self.g_loss_history))
        g_mean = np.mean(list(self.g_loss_history))
        
        # Converged if std < 5% of mean
        return g_std < 0.05 * g_mean
    
    def print_epoch_summary(
        self,
        epoch: int,
        total_epochs: int,
        losses: Dict[str, float],
        checkpoint_saved: bool = False
    ):
        """Print formatted epoch summary."""
        print("\n" + "=" * 70)
        print(f"EPOCH {epoch}/{total_epochs} SUMMARY")
        print("=" * 70)
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\nLosses:")
        print(f"  Generator:     {losses.get('g_loss', 0):.6f}")
        print(f"  Discriminator: {losses.get('d_loss', 0):.6f}")
        print(f"  Cover:         {losses.get('cover_loss', 0):.6f}")
        print(f"  Secret:        {losses.get('secret_loss', 0):.6f}")
        
        if 'ssim_loss' in losses:
            print(f"  SSIM:          {losses.get('ssim_loss', 0):.6f}")
        
        if checkpoint_saved:
            print(f"\n✅ Checkpoint saved")
        
        # Print status
        status = self.update(
            epoch,
            losses.get('g_loss', 0),
            losses.get('d_loss', 0),
            losses.get('cover_loss', 0),
            losses.get('secret_loss', 0)
        )
        
        print(f"\nStatus: {status['status']}")
        if status.get('converged'):
            print("⚠️ Training appears converged - diminishing returns expected")
        
        if status.get('anomalies'):
            print("\n⚠️ Recent anomalies:")
            for anomaly in status['anomalies']:
                print(f"  Epoch {anomaly['epoch']}: {anomaly['details']}")
        
        print("=" * 70)
        
        return status


class GPUMonitor:
    """Monitor GPU memory usage during training."""
    
    def __init__(self, alert_threshold: float = 0.85):
        """
        Args:
            alert_threshold: Trigger alert if memory usage exceeds this fraction
        """
        self.alert_threshold = alert_threshold
        self.peak_memory = 0
        self.oom_count = 0
    
    def check(self) -> Dict:
        """
        Check current GPU memory status.
        
        Returns:
            Dict with memory stats and alerts
        """
        if not torch.cuda.is_available():
            return {'available': False}
        
        props = torch.cuda.get_device_properties(0)
        total = props.total_memory / 1e9
        allocated = torch.cuda.memory_allocated(0) / 1e9
        reserved = torch.cuda.memory_reserved(0) / 1e9
        
        usage_fraction = reserved / total
        self.peak_memory = max(self.peak_memory, reserved)
        
        alert = usage_fraction > self.alert_threshold
        
        return {
            'available': True,
            'total_gb': total,
            'allocated_gb': allocated,
            'reserved_gb': reserved,
            'usage_fraction': usage_fraction,
            'peak_gb': self.peak_memory,
            'alert': alert,
            'alert_message': f"High GPU memory usage: {usage_fraction*100:.1f}%" if alert else None
        }
    
    def print_status(self):
        """Print current GPU memory status."""
        status = self.check()
        if not status['available']:
            print("GPU not available")
            return
        
        print(f"\nGPU Memory: {status['reserved_gb']:.2f}/{status['total_gb']:.2f} GB "
              f"({status['usage_fraction']*100:.1f}%)")
        
        if status['alert']:
            print(f"⚠️ {status['alert_message']}")


if __name__ == '__main__':
    # Test monitor
    monitor = TrainingMonitor(warmup_batches=100)
    gpu_monitor = GPUMonitor()
    
    # Simulate training
    for epoch in range(1, 6):
        losses = {
            'g_loss': 0.05 + np.random.randn() * 0.01,
            'd_loss': 0.1 + np.random.randn() * 0.01,
            'cover_loss': 0.01 + np.random.randn() * 0.001,
            'secret_loss': 0.02 + np.random.randn() * 0.002
        }
        
        status = monitor.print_epoch_summary(epoch, 10, losses, checkpoint_saved=True)
        gpu_monitor.print_status()
        
        if status.get('should_stop'):
            print(f"\n❌ Training aborted: {status['reason']}")
            break
