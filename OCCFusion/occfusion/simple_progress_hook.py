import time
from typing import Optional, Dict, Any
from mmengine.hooks import Hook
from mmengine.registry import HOOKS
from tqdm import tqdm


@HOOKS.register_module()
class SimpleProgressBarHook(Hook):
    """Simplified progress bar hook to avoid formatting errors.
    
    Args:
        bar_width (int): Width of the progress bar. Default: 80.
        show_eta (bool): Whether to show estimated time of arrival. Default: True.
    """
    
    def __init__(self, 
                 bar_width: int = 80,
                 show_eta: bool = True):
        self.bar_width = bar_width
        self.show_eta = show_eta
        
        # Progress tracking
        self.epoch_pbar = None
        self.iter_pbar = None
        self.start_time = None
        self.epoch_start_time = None

    def before_run(self, runner) -> None:
        """Initialize progress tracking before training starts."""
        self.start_time = time.time()
        total_epochs = runner.max_epochs
        
        print(f"\n🚀 Starting Training: {total_epochs} epochs")
        print(f"📊 Dataset: {len(runner.train_dataloader)} batches per epoch")
        print("-" * 80)

    def before_train_epoch(self, runner) -> None:
        """Initialize epoch progress bar."""
        self.epoch_start_time = time.time()
        current_epoch = runner.epoch + 1
        total_epochs = runner.max_epochs
        
        # Close previous progress bar if exists
        if self.iter_pbar is not None:
            self.iter_pbar.close()
            
        # Create new progress bar for this epoch
        total_iters = len(runner.train_dataloader)
        desc = f"Epoch {current_epoch:2d}/{total_epochs}"
        
        self.iter_pbar = tqdm(
            total=total_iters,
            desc=desc,
            ncols=self.bar_width,
            unit="batch",
            leave=True,
            dynamic_ncols=True
        )

    def after_train_iter(self, 
                        runner,
                        batch_idx: int,
                        data_batch: Optional[Any] = None,
                        outputs: Optional[Dict] = None) -> None:
        """Update progress bar after each training iteration."""
        
        if self.iter_pbar is None:
            return
            
        # Simple postfix - just show iteration number
        postfix_dict = {}
        
        # Try to get the most basic loss info
        try:
            if hasattr(runner, 'log_buffer') and runner.log_buffer.output:
                for key, value in runner.log_buffer.output.items():
                    if 'loss' in key.lower() and isinstance(value, (int, float)):
                        # Only show the first loss found
                        postfix_dict['loss'] = f"{value:.4f}"
                        break
        except:
            pass
        
        # Update progress bar
        try:
            if postfix_dict:
                self.iter_pbar.set_postfix(postfix_dict)
            self.iter_pbar.update(1)
        except:
            # If even this fails, just update without postfix
            self.iter_pbar.update(1)

    def after_train_epoch(self, runner) -> None:
        """Clean up after epoch completion."""
        if self.iter_pbar is not None:
            self.iter_pbar.close()
            self.iter_pbar = None
        
        # Print epoch summary
        epoch_time = time.time() - self.epoch_start_time
        current_epoch = runner.epoch + 1
        total_epochs = runner.max_epochs
        
        print(f"\n✅ Epoch {current_epoch}/{total_epochs} completed in {epoch_time:.1f}s")
        
        # Estimate remaining time
        if self.show_eta and current_epoch < total_epochs:
            elapsed_total = time.time() - self.start_time
            avg_epoch_time = elapsed_total / current_epoch
            remaining_epochs = total_epochs - current_epoch
            eta_seconds = remaining_epochs * avg_epoch_time
            
            eta_hours = int(eta_seconds // 3600)
            eta_minutes = int((eta_seconds % 3600) // 60)
            
            if eta_hours > 0:
                eta_str = f"{eta_hours}h {eta_minutes}m"
            else:
                eta_str = f"{eta_minutes}m"
                
            print(f"⏰ ETA: {eta_str} remaining")
        
        print("-" * 80)

    def before_val_epoch(self, runner) -> None:
        """Handle validation epoch start."""
        if hasattr(runner, 'val_dataloader') and runner.val_dataloader:
            total_val_iters = len(runner.val_dataloader)
            desc = f"Validating"
            
            self.val_pbar = tqdm(
                total=total_val_iters,
                desc=desc,
                ncols=self.bar_width,
                unit="batch",
                leave=False,
                colour='green'
            )

    def after_val_iter(self, 
                      runner,
                      batch_idx: int,
                      data_batch: Optional[Any] = None,
                      outputs: Optional[Dict] = None) -> None:
        """Update validation progress."""
        if hasattr(self, 'val_pbar') and self.val_pbar is not None:
            self.val_pbar.update(1)

    def after_val_epoch(self, runner, metrics: Optional[Dict] = None) -> None:
        """Clean up after validation."""
        if hasattr(self, 'val_pbar') and self.val_pbar is not None:
            self.val_pbar.close()
            delattr(self, 'val_pbar')
        
        # Print validation results
        if metrics:
            print(f"🎯 Validation completed")

    def after_run(self, runner) -> None:
        """Print final training summary."""
        total_time = time.time() - self.start_time
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        
        print("\n" + "="*80)
        print(f"🎉 Training Completed!")
        print(f"⏱️  Total time: {hours}h {minutes}m")
        print("="*80)