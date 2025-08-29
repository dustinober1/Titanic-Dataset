#!/usr/bin/env python3
"""
Docker Management Script for Titanic ML API
==========================================

Utility script for managing Docker containers, images, and deployments
for the Titanic ML project.

Features:
- Build and manage containers
- Deploy to different environments
- Health checks and monitoring
- Log management
- Backup and restore operations

Usage:
    python scripts/docker_management.py [command] [options]

Author: Enhanced Titanic ML Framework
"""

import argparse
import subprocess
import sys
import time
import json
import os
from pathlib import Path
from typing import List, Dict, Any
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DockerManager:
    """Docker container and image management for Titanic ML"""
    
    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path(__file__).parent.parent
        self.compose_files = {
            'prod': self.project_root / 'docker-compose.yml',
            'dev': self.project_root / 'docker-compose.dev.yml'
        }
    
    def run_command(self, command: List[str], capture_output: bool = False) -> subprocess.CompletedProcess:
        """Run a shell command"""
        logger.info(f"Running command: {' '.join(command)}")
        
        try:
            if capture_output:
                result = subprocess.run(command, capture_output=True, text=True, check=True)
            else:
                result = subprocess.run(command, check=True)
            return result
        except subprocess.CalledProcessError as e:
            logger.error(f"Command failed: {e}")
            if capture_output and e.stdout:
                logger.error(f"STDOUT: {e.stdout}")
            if capture_output and e.stderr:
                logger.error(f"STDERR: {e.stderr}")
            raise
    
    def build_images(self, environment: str = 'prod', no_cache: bool = False):
        """Build Docker images"""
        logger.info(f"🏗️  Building {environment} images...")
        
        compose_file = self.compose_files[environment]
        if not compose_file.exists():
            raise FileNotFoundError(f"Docker compose file not found: {compose_file}")
        
        command = ['docker-compose', '-f', str(compose_file), 'build']
        if no_cache:
            command.append('--no-cache')
        
        self.run_command(command)
        logger.info("✅ Images built successfully")
    
    def start_services(self, environment: str = 'prod', detached: bool = True, services: List[str] = None):
        """Start Docker services"""
        logger.info(f"🚀 Starting {environment} services...")
        
        compose_file = self.compose_files[environment]
        command = ['docker-compose', '-f', str(compose_file), 'up']
        
        if detached:
            command.append('-d')
        
        if services:
            command.extend(services)
        
        self.run_command(command)
        
        if detached:
            logger.info("✅ Services started in background")
        else:
            logger.info("✅ Services started")
    
    def stop_services(self, environment: str = 'prod'):
        """Stop Docker services"""
        logger.info(f"🛑 Stopping {environment} services...")
        
        compose_file = self.compose_files[environment]
        command = ['docker-compose', '-f', str(compose_file), 'down']
        
        self.run_command(command)
        logger.info("✅ Services stopped")
    
    def restart_services(self, environment: str = 'prod', services: List[str] = None):
        """Restart Docker services"""
        logger.info(f"🔄 Restarting {environment} services...")
        
        compose_file = self.compose_files[environment]
        command = ['docker-compose', '-f', str(compose_file), 'restart']
        
        if services:
            command.extend(services)
        
        self.run_command(command)
        logger.info("✅ Services restarted")
    
    def check_health(self, environment: str = 'prod') -> Dict[str, Any]:
        """Check health of running containers"""
        logger.info(f"🔍 Checking health of {environment} containers...")
        
        compose_file = self.compose_files[environment]
        
        # Get container status
        result = self.run_command([
            'docker-compose', '-f', str(compose_file), 'ps', '--format', 'json'
        ], capture_output=True)
        
        containers = []
        if result.stdout:
            for line in result.stdout.strip().split('\n'):
                if line:
                    try:
                        containers.append(json.loads(line))
                    except json.JSONDecodeError:
                        # Handle non-JSON output format
                        pass
        
        # Check API health endpoint
        api_healthy = self.check_api_health()
        
        health_status = {
            'containers': containers,
            'api_healthy': api_healthy,
            'timestamp': time.time()
        }
        
        logger.info(f"📊 Health check completed: {len(containers)} containers")
        return health_status
    
    def check_api_health(self) -> bool:
        """Check if API is responding"""
        try:
            import requests
            response = requests.get('http://localhost:8000/health', timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"API health check failed: {e}")
            return False
    
    def view_logs(self, environment: str = 'prod', service: str = None, tail: int = 100, follow: bool = False):
        """View container logs"""
        logger.info(f"📋 Viewing logs for {environment} environment...")
        
        compose_file = self.compose_files[environment]
        command = ['docker-compose', '-f', str(compose_file), 'logs']
        
        if tail:
            command.extend(['--tail', str(tail)])
        
        if follow:
            command.append('-f')
        
        if service:
            command.append(service)
        
        self.run_command(command)
    
    def scale_service(self, environment: str = 'prod', service: str = 'titanic-api', replicas: int = 2):
        """Scale a service to specified number of replicas"""
        logger.info(f"📈 Scaling {service} to {replicas} replicas...")
        
        compose_file = self.compose_files[environment]
        command = [
            'docker-compose', '-f', str(compose_file), 
            'up', '-d', '--scale', f'{service}={replicas}'
        ]
        
        self.run_command(command)
        logger.info(f"✅ {service} scaled to {replicas} replicas")
    
    def backup_data(self, backup_dir: Path = None):
        """Backup persistent data"""
        backup_dir = backup_dir or self.project_root / 'backups'
        backup_dir.mkdir(exist_ok=True)
        
        timestamp = int(time.time())
        
        logger.info("💾 Creating data backup...")
        
        # Backup volumes
        volumes_to_backup = [
            'postgres-data',
            'redis-data', 
            'prometheus-data',
            'grafana-data'
        ]
        
        for volume in volumes_to_backup:
            try:
                backup_file = backup_dir / f'{volume}_backup_{timestamp}.tar.gz'
                
                command = [
                    'docker', 'run', '--rm',
                    '-v', f'{volume}:/data:ro',
                    '-v', f'{backup_dir}:/backup',
                    'alpine:latest',
                    'tar', 'czf', f'/backup/{backup_file.name}', '-C', '/data', '.'
                ]
                
                self.run_command(command)
                logger.info(f"✅ Backed up {volume} to {backup_file}")
                
            except Exception as e:
                logger.error(f"❌ Failed to backup {volume}: {e}")
        
        logger.info("💾 Backup completed")
    
    def restore_data(self, backup_file: Path, volume_name: str):
        """Restore data from backup"""
        if not backup_file.exists():
            raise FileNotFoundError(f"Backup file not found: {backup_file}")
        
        logger.info(f"🔄 Restoring {volume_name} from {backup_file}...")
        
        command = [
            'docker', 'run', '--rm',
            '-v', f'{volume_name}:/data',
            '-v', f'{backup_file.parent}:/backup:ro',
            'alpine:latest',
            'tar', 'xzf', f'/backup/{backup_file.name}', '-C', '/data'
        ]
        
        self.run_command(command)
        logger.info(f"✅ Restored {volume_name}")
    
    def deploy_production(self, build: bool = True, migrate: bool = True):
        """Deploy to production environment"""
        logger.info("🚀 Starting production deployment...")
        
        try:
            # Stop existing services
            self.stop_services('prod')
            
            # Build new images if requested
            if build:
                self.build_images('prod', no_cache=True)
            
            # Start services
            self.start_services('prod')
            
            # Wait for services to be ready
            logger.info("⏳ Waiting for services to be ready...")
            time.sleep(30)
            
            # Health check
            health = self.check_health('prod')
            if health['api_healthy']:
                logger.info("✅ Production deployment successful!")
            else:
                logger.error("❌ API health check failed after deployment")
                raise Exception("Deployment health check failed")
                
        except Exception as e:
            logger.error(f"❌ Production deployment failed: {e}")
            raise
    
    def cleanup(self, environment: str = 'prod', remove_volumes: bool = False):
        """Clean up containers and images"""
        logger.info(f"🧹 Cleaning up {environment} environment...")
        
        compose_file = self.compose_files[environment]
        
        # Stop and remove containers
        command = ['docker-compose', '-f', str(compose_file), 'down']
        if remove_volumes:
            command.append('-v')
        
        self.run_command(command)
        
        # Remove dangling images
        try:
            self.run_command(['docker', 'image', 'prune', '-f'])
        except:
            pass
        
        logger.info("✅ Cleanup completed")
    
    def get_resource_usage(self, environment: str = 'prod') -> Dict[str, Any]:
        """Get resource usage statistics"""
        logger.info("📊 Getting resource usage statistics...")
        
        compose_file = self.compose_files[environment]
        
        # Get container stats
        try:
            result = self.run_command([
                'docker-compose', '-f', str(compose_file), 'top'
            ], capture_output=True)
            
            stats_output = result.stdout if result.stdout else ""
            
        except Exception as e:
            logger.warning(f"Failed to get container stats: {e}")
            stats_output = ""
        
        # Get system stats
        try:
            system_result = self.run_command([
                'docker', 'system', 'df'
            ], capture_output=True)
            
            system_stats = system_result.stdout if system_result.stdout else ""
            
        except Exception as e:
            logger.warning(f"Failed to get system stats: {e}")
            system_stats = ""
        
        return {
            'container_stats': stats_output,
            'system_stats': system_stats,
            'timestamp': time.time()
        }


def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(
        description='Docker Management Script for Titanic ML API',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python docker_management.py build --env prod
  python docker_management.py start --env dev --detached
  python docker_management.py logs --service titanic-api --follow
  python docker_management.py health --env prod
  python docker_management.py deploy --build
        """
    )
    
    parser.add_argument('command', choices=[
        'build', 'start', 'stop', 'restart', 'logs', 'health', 
        'scale', 'backup', 'restore', 'deploy', 'cleanup', 'stats'
    ], help='Command to execute')
    
    parser.add_argument('--env', choices=['prod', 'dev'], default='prod',
                       help='Environment (prod/dev)')
    
    parser.add_argument('--service', help='Specific service name')
    
    parser.add_argument('--detached', '-d', action='store_true',
                       help='Run in detached mode')
    
    parser.add_argument('--follow', '-f', action='store_true',
                       help='Follow logs output')
    
    parser.add_argument('--tail', type=int, default=100,
                       help='Number of lines to show from logs')
    
    parser.add_argument('--build', action='store_true',
                       help='Build images before starting')
    
    parser.add_argument('--no-cache', action='store_true',
                       help='Build without cache')
    
    parser.add_argument('--replicas', type=int, default=2,
                       help='Number of replicas for scaling')
    
    parser.add_argument('--backup-dir', type=Path,
                       help='Directory for backups')
    
    parser.add_argument('--backup-file', type=Path,
                       help='Backup file to restore from')
    
    parser.add_argument('--volume', help='Volume name for restore')
    
    parser.add_argument('--remove-volumes', action='store_true',
                       help='Remove volumes during cleanup')
    
    args = parser.parse_args()
    
    # Initialize Docker manager
    manager = DockerManager()
    
    try:
        if args.command == 'build':
            manager.build_images(args.env, args.no_cache)
        
        elif args.command == 'start':
            services = [args.service] if args.service else None
            manager.start_services(args.env, args.detached, services)
        
        elif args.command == 'stop':
            manager.stop_services(args.env)
        
        elif args.command == 'restart':
            services = [args.service] if args.service else None
            manager.restart_services(args.env, services)
        
        elif args.command == 'logs':
            manager.view_logs(args.env, args.service, args.tail, args.follow)
        
        elif args.command == 'health':
            health = manager.check_health(args.env)
            print(json.dumps(health, indent=2))
        
        elif args.command == 'scale':
            if not args.service:
                args.service = 'titanic-api'
            manager.scale_service(args.env, args.service, args.replicas)
        
        elif args.command == 'backup':
            manager.backup_data(args.backup_dir)
        
        elif args.command == 'restore':
            if not args.backup_file or not args.volume:
                parser.error("--backup-file and --volume are required for restore")
            manager.restore_data(args.backup_file, args.volume)
        
        elif args.command == 'deploy':
            manager.deploy_production(args.build)
        
        elif args.command == 'cleanup':
            manager.cleanup(args.env, args.remove_volumes)
        
        elif args.command == 'stats':
            stats = manager.get_resource_usage(args.env)
            print("Container Stats:")
            print(stats['container_stats'])
            print("\nSystem Stats:")
            print(stats['system_stats'])
        
    except Exception as e:
        logger.error(f"❌ Command failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()