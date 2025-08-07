import time
from typing import Optional, Dict, Any
from mmengine.hooks import Hook
from mmengine.registry import HOOKS
from tqdm import tqdm


@HOOKS.register_module()
class ProgressBarHook(Hook):
    """Hook to display training progress with tqdm progress bar.
    
    Args:
        bar_width (int): Width of the progress bar. Default: 80.
        show_eta (bool): Whether to show estimated time of arrival. Default: True.
        show_loss (bool): Whether to show current loss values. Default: True.
        update_interval (int): Update interval for the progress bar. Default: 1.
    """
    
    def __init__(self, 
                 bar_width: int = 80,
                 show_eta: bool = True,
                 show_loss: bool = True,
                 update_interval: int = 1):
        self.bar_width = bar_width
        self.show_eta = show_eta
        self.show_loss = show_loss
        self.update_interval = update_interval
        
        # Progress tracking
        self.epoch_pbar = None
        self.iter_pbar = None
        self.start_time = None
        self.epoch_start_time = None
        
        # Loss tracking
        self.running_losses = {}
        self.loss_window_size = 50
        
    def before_run(self, runner) -> None:
        """Initialize progress tracking before training starts."""
        self.start_time = time.time()
        total_epochs = runner.max_epochs
        
        print(f"\n🚀 Starting Training: {total_epochs} epochs")
        print(f"📊 Dataset: {len(runner.train_dataloader)} batches per epoch")
        print(f"⚙️  Model: {runner.model.__class__.__name__}")
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
        
        # Reset running losses for new epoch
        self.running_losses = {}

    def after_train_iter(self, 
                        runner,
                        batch_idx: int,
                        data_batch: Optional[Any] = None,
                        outputs: Optional[Dict] = None) -> None:
        """Update progress bar after each training iteration."""
        
        if self.iter_pbar is None:
            return
            
        # Update losses
        if hasattr(runner, 'log_buffer') and runner.log_buffer.output:
            current_losses = {}
            for key, value in runner.log_buffer.output.items():
                if 'loss' in key.lower():
                    if key not in self.running_losses:
                        self.running_losses[key] = []
                    
                    # Keep a rolling window of losses
                    self.running_losses[key].append(float(value))
                    if len(self.running_losses[key]) > self.loss_window_size:
                        self.running_losses[key].pop(0)
                    
                    # Calculate moving average
                    avg_loss = sum(self.running_losses[key]) / len(self.running_losses[key])
                    current_losses[key] = avg_loss

        # Create postfix string
        postfix_dict = {}
        
        # Add learning rate
        try:
            if hasattr(runner, 'optim_wrapper') and hasattr(runner.optim_wrapper, 'get_lr'):
                lr = runner.optim_wrapper.get_lr()
                if isinstance(lr, dict):
                    lr_val = list(lr.values())[0] if lr else 0
                elif isinstance(lr, list):
                    lr_val = lr[0] if lr else 0
                else:
                    lr_val = lr if lr is not None else 0
                
                # Ensure lr_val is a number
                if isinstance(lr_val, (int, float)):
                    postfix_dict['lr'] = f"{lr_val:.2e}"
                else:
                    # Handle case where lr_val might still be a list or other type
                    try:
                        lr_num = float(lr_val)
                        postfix_dict['lr'] = f"{lr_num:.2e}"
                    except (ValueError, TypeError):
                        postfix_dict['lr'] = "N/A"
        except Exception:
            # Fallback - don't show learning rate if there's any issue
            pass
        
        # Add losses (limit to 2-3 most important ones)
        if self.show_loss and current_losses:
            try:
                loss_items = list(current_losses.items())[:3]  # Show max 3 losses
                for key, value in loss_items:
                    try:
                        # Shorten loss names for display
                        short_key = key.replace('loss', '').replace('_loss', '').replace('level0_', '')
                        if short_key == '':
                            short_key = 'total'
                        
                        # Ensure value is a number
                        if isinstance(value, (int, float)):
                            postfix_dict[short_key] = f"{value:.4f}"
                        else:
                            # Try to convert to float
                            val_num = float(value)
                            postfix_dict[short_key] = f"{val_num:.4f}"
                    except (ValueError, TypeError):
                        # Skip this loss if it can't be formatted
                        continue
            except Exception:
                # Skip loss display if there's any issue
                pass
        
        # Add memory usage if available
        try:
            if hasattr(runner, 'log_buffer') and runner.log_buffer.output and 'memory' in runner.log_buffer.output:
                memory_mb = runner.log_buffer.output['memory']
                if isinstance(memory_mb, (int, float)):
                    postfix_dict['mem'] = f"{memory_mb:.0f}MB"
        except Exception:
            # Skip memory display if there's any issue
            pass
        
        # Update progress bar
        self.iter_pbar.set_postfix(postfix_dict)
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
        
        # Print final losses for this epoch
        if self.running_losses:
            loss_str = " | ".join([f"{k}: {sum(v)/len(v):.4f}" for k, v in self.running_losses.items()])
            print(f"📈 Final losses: {loss_str}")
        
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
            metrics_str = " | ".join([f"{k}: {v:.4f}" if isinstance(v, (int, float)) else f"{k}: {v}" 
                                    for k, v in metrics.items() if not k.startswith('_')])
            print(f"🎯 Validation: {metrics_str}")

    def after_run(self, runner) -> None:
        """Print final training summary."""
        total_time = time.time() - self.start_time
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        
        print("\n" + "="*80)
        print(f"🎉 Training Completed!")
        print(f"⏱️  Total time: {hours}h {minutes}m")
        print(f"📁 Work directory: {runner.work_dir}")
        print("="*80)