"""Training pipeline orchestrator for sequential environment execution."""

import os
import json
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging

from src.utils.config_loader import load_config
from src.utils.logger import setup_logger
from src.training.trainer import train_agent


class TrainingPipeline:
    """
    Orchestrates sequential training across multiple environments.
    
    Features:
    - Sequential execution of training sessions
    - Checkpoint management and transfer learning
    - State persistence across sessions
    - AWS S3 integration for distributed training
    - Comprehensive logging and monitoring
    """
    
    def __init__(
        self,
        pipeline_config_path: str,
        output_dir: str = "./pipeline_output",
        use_s3: bool = False,
        s3_bucket: Optional[str] = None
    ):
        """
        Initialize training pipeline.
        
        Args:
            pipeline_config_path: Path to pipeline configuration file
            output_dir: Local directory for pipeline outputs
            use_s3: Whether to use S3 for checkpoint storage
            s3_bucket: S3 bucket name (required if use_s3=True)
        """
        self.pipeline_config = load_config(pipeline_config_path)
        self.output_dir = Path(output_dir)
        self.use_s3 = use_s3
        self.s3_bucket = s3_bucket
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup pipeline logger
        self.logger = setup_logger(
            'pipeline',
            log_file=str(self.output_dir / 'pipeline.log')
        )
        
        # Initialize state tracking
        self.state_file = self.output_dir / 'pipeline_state.json'
        self.state = self._load_state()
        
        # AWS S3 setup if enabled
        if self.use_s3:
            if not self.s3_bucket:
                raise ValueError("s3_bucket required when use_s3=True")
            self._setup_s3()
        
        self.logger.info(f"Pipeline initialized: {pipeline_config_path}")
        self.logger.info(f"Output directory: {self.output_dir}")
        
    def _setup_s3(self):
        """Setup AWS S3 client for checkpoint management."""
        try:
            import boto3
            self.s3_client = boto3.client('s3')
            self.logger.info(f"S3 client initialized for bucket: {self.s3_bucket}")
        except ImportError:
            self.logger.error("boto3 not installed. Install with: pip install boto3")
            raise
        except Exception as e:
            self.logger.error(f"Failed to initialize S3 client: {e}")
            raise
    
    def _load_state(self) -> Dict[str, Any]:
        """Load pipeline state from disk."""
        if self.state_file.exists():
            with open(self.state_file, 'r') as f:
                return json.load(f)
        return {
            'sessions_completed': [],
            'current_session': None,
            'started_at': None,
            'last_updated': None,
            'checkpoints': {}
        }
    
    def _save_state(self):
        """Save pipeline state to disk."""
        self.state['last_updated'] = datetime.now().isoformat()
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)
        
        # Upload to S3 if enabled
        if self.use_s3:
            self._upload_to_s3(
                self.state_file,
                f"pipeline_states/{self.state_file.name}"
            )
    
    def _upload_to_s3(self, local_path: Path, s3_key: str):
        """Upload file to S3."""
        try:
            self.s3_client.upload_file(
                str(local_path),
                self.s3_bucket,
                s3_key
            )
            self.logger.info(f"Uploaded to S3: s3://{self.s3_bucket}/{s3_key}")
        except Exception as e:
            self.logger.error(f"Failed to upload to S3: {e}")
            raise
    
    def _download_from_s3(self, s3_key: str, local_path: Path):
        """Download file from S3."""
        try:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            self.s3_client.download_file(
                self.s3_bucket,
                s3_key,
                str(local_path)
            )
            self.logger.info(f"Downloaded from S3: {s3_key}")
        except Exception as e:
            self.logger.error(f"Failed to download from S3: {e}")
            raise
    
    def _prepare_session_config(
        self,
        session: Dict[str, Any],
        checkpoint_path: Optional[str] = None
    ) -> Path:
        """
        Prepare training configuration for a session.
        
        Args:
            session: Session configuration dict
            checkpoint_path: Optional checkpoint to resume from
            
        Returns:
            Path to generated config file
        """
        # Load base config
        base_config = load_config(session['config'])
        
        # Override with session-specific settings
        if 'overrides' in session:
            base_config.update(session['overrides'])
        
        # Setup session-specific paths
        session_name = session['name']
        session_output = self.output_dir / session_name
        session_output.mkdir(parents=True, exist_ok=True)
        
        base_config['paths']['models'] = str(session_output / 'models')
        base_config['paths']['logs'] = str(session_output / 'logs')
        base_config['paths']['tensorboard'] = str(session_output / 'runs')
        
        # Add checkpoint if provided
        if checkpoint_path:
            base_config['checkpoint'] = checkpoint_path
        
        # Save session config
        config_path = session_output / 'training_config.yaml'
        import yaml
        with open(config_path, 'w') as f:
            yaml.dump(base_config, f, default_flow_style=False)
        
        return config_path
    
    def _get_latest_checkpoint(self, session_name: str) -> Optional[str]:
        """Get path to latest checkpoint for a session."""
        session_output = self.output_dir / session_name / 'models'
        
        if not session_output.exists():
            return None
        
        # Look for best_model first, then final_model
        best_model = session_output / 'best_model.zip'
        final_model = session_output / 'final_model.zip'
        
        if best_model.exists():
            return str(best_model)
        elif final_model.exists():
            return str(final_model)
        
        return None
    
    def run_session(
        self,
        session: Dict[str, Any],
        checkpoint_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run a single training session.
        
        Args:
            session: Session configuration
            checkpoint_path: Optional checkpoint to resume from
            
        Returns:
            Session results dict
        """
        session_name = session['name']
        self.logger.info(f"{'='*80}")
        self.logger.info(f"Starting Training Session: {session_name}")
        self.logger.info(f"{'='*80}")
        
        # Update state
        self.state['current_session'] = session_name
        if not self.state['started_at']:
            self.state['started_at'] = datetime.now().isoformat()
        self._save_state()
        
        # Prepare config
        config_path = self._prepare_session_config(session, checkpoint_path)
        self.logger.info(f"Session config: {config_path}")
        
        # Run training
        try:
            start_time = datetime.now()
            
            agent = train_agent(
                config_path=str(config_path),
                checkpoint_path=checkpoint_path,
                total_timesteps=session.get('timesteps')
            )
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Get checkpoint path
            final_checkpoint = self._get_latest_checkpoint(session_name)
            
            # Record results
            results = {
                'session': session_name,
                'status': 'completed',
                'started_at': start_time.isoformat(),
                'completed_at': end_time.isoformat(),
                'duration_seconds': duration,
                'checkpoint': final_checkpoint
            }
            
            # Update state
            self.state['sessions_completed'].append(session_name)
            self.state['checkpoints'][session_name] = final_checkpoint
            self.state['current_session'] = None
            self._save_state()
            
            # Upload checkpoint to S3 if enabled
            if self.use_s3 and final_checkpoint:
                checkpoint_s3_key = f"checkpoints/{session_name}/best_model.zip"
                self._upload_to_s3(Path(final_checkpoint), checkpoint_s3_key)
                results['s3_checkpoint'] = f"s3://{self.s3_bucket}/{checkpoint_s3_key}"
            
            self.logger.info(f"Session '{session_name}' completed successfully")
            self.logger.info(f"Duration: {duration:.2f} seconds")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Session '{session_name}' failed: {e}", exc_info=True)
            
            # Update state
            self.state['current_session'] = None
            self._save_state()
            
            return {
                'session': session_name,
                'status': 'failed',
                'error': str(e)
            }
    
    def run(self, start_from: Optional[str] = None, resume: bool = False):
        """
        Run the complete training pipeline.
        
        Args:
            start_from: Session name to start from (skips earlier sessions)
            resume: Whether to resume from last checkpoint
        """
        self.logger.info("="*80)
        self.logger.info("TRAINING PIPELINE STARTED")
        self.logger.info("="*80)
        
        sessions = self.pipeline_config['sessions']
        
        # Determine starting point
        start_idx = 0
        if start_from:
            try:
                start_idx = next(
                    i for i, s in enumerate(sessions)
                    if s['name'] == start_from
                )
                self.logger.info(f"Starting from session: {start_from}")
            except StopIteration:
                self.logger.error(f"Session '{start_from}' not found in pipeline")
                return
        elif resume and self.state['current_session']:
            # Resume from interrupted session
            try:
                start_idx = next(
                    i for i, s in enumerate(sessions)
                    if s['name'] == self.state['current_session']
                )
                self.logger.info(f"Resuming from session: {self.state['current_session']}")
            except StopIteration:
                self.logger.warning("Could not find session to resume, starting from beginning")
        
        # Execute pipeline
        results = []
        previous_checkpoint = None
        
        for idx, session in enumerate(sessions[start_idx:], start=start_idx):
            session_name = session['name']
            
            # Skip completed sessions unless forcing restart
            if session_name in self.state['sessions_completed'] and not resume:
                self.logger.info(f"Skipping completed session: {session_name}")
                previous_checkpoint = self.state['checkpoints'].get(session_name)
                continue
            
            # Determine checkpoint source
            checkpoint = None
            if session.get('use_previous_checkpoint', False) and previous_checkpoint:
                checkpoint = previous_checkpoint
                self.logger.info(f"Using checkpoint from previous session: {checkpoint}")
            elif session.get('checkpoint'):
                checkpoint = session['checkpoint']
                self.logger.info(f"Using specified checkpoint: {checkpoint}")
            
            # Run session
            result = self.run_session(session, checkpoint)
            results.append(result)
            
            # Update previous checkpoint for next session
            if result['status'] == 'completed':
                previous_checkpoint = result['checkpoint']
            else:
                # Stop pipeline on failure unless continue_on_error is set
                if not self.pipeline_config.get('continue_on_error', False):
                    self.logger.error("Pipeline stopped due to session failure")
                    break
        
        # Generate final report
        self._generate_report(results)
        
        self.logger.info("="*80)
        self.logger.info("TRAINING PIPELINE COMPLETED")
        self.logger.info("="*80)
    
    def _generate_report(self, results: List[Dict[str, Any]]):
        """Generate pipeline execution report."""
        report_path = self.output_dir / 'pipeline_report.json'
        
        report = {
            'pipeline': self.pipeline_config.get('name', 'Unnamed Pipeline'),
            'started_at': self.state['started_at'],
            'completed_at': datetime.now().isoformat(),
            'sessions': results,
            'summary': {
                'total_sessions': len(results),
                'completed': sum(1 for r in results if r['status'] == 'completed'),
                'failed': sum(1 for r in results if r['status'] == 'failed'),
                'total_duration': sum(
                    r.get('duration_seconds', 0) for r in results
                )
            }
        }
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Pipeline report saved to: {report_path}")
        
        # Upload report to S3 if enabled
        if self.use_s3:
            self._upload_to_s3(
                report_path,
                f"reports/pipeline_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )


def main():
    """CLI entry point for pipeline execution."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Execute training pipeline with sequential environments"
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to pipeline configuration file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./pipeline_output',
        help='Directory for pipeline outputs'
    )
    parser.add_argument(
        '--start-from',
        type=str,
        default=None,
        help='Session name to start from (skips earlier sessions)'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from last interrupted session'
    )
    parser.add_argument(
        '--use-s3',
        action='store_true',
        help='Use S3 for checkpoint storage'
    )
    parser.add_argument(
        '--s3-bucket',
        type=str,
        default=None,
        help='S3 bucket name for checkpoint storage'
    )
    
    args = parser.parse_args()
    
    # Create and run pipeline
    pipeline = TrainingPipeline(
        pipeline_config_path=args.config,
        output_dir=args.output_dir,
        use_s3=args.use_s3,
        s3_bucket=args.s3_bucket
    )
    
    pipeline.run(
        start_from=args.start_from,
        resume=args.resume
    )


if __name__ == '__main__':
    main()
