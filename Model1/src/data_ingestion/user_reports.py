import asyncio
import json
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from dataclasses import asdict
from ..data_ingestion import DataSource, RawReport
from config import config

class UserReportsSource(DataSource):
    """User reports data source for citizen-submitted hazard reports"""
    
    def __init__(self):
        self.pending_reports = []
        self.processed_reports = set()
    
    async def setup(self) -> None:
        """Setup user reports source"""
        print("User reports source setup completed")
    
    async def fetch_data(self, **kwargs) -> List[RawReport]:
        """Fetch pending user reports"""
        # In a real implementation, this would fetch from a queue or database
        reports = self.pending_reports.copy()
        self.pending_reports.clear()
        
        print(f"Fetched {len(reports)} user reports")
        return reports
    
    def submit_report(self, report_data: Dict[str, Any]) -> str:
        """Submit a new user report (called by API endpoints)"""
        try:
            # Validate required fields
            required_fields = ['content', 'timestamp']
            for field in required_fields:
                if field not in report_data:
                    raise ValueError(f"Missing required field: {field}")
            
            # Generate unique ID
            report_id = f"user_{hash(str(report_data))}_{int(datetime.now().timestamp())}"
            
            # Parse timestamp
            if isinstance(report_data['timestamp'], str):
                timestamp = datetime.fromisoformat(report_data['timestamp'].replace('Z', '+00:00'))
            else:
                timestamp = report_data['timestamp']
            
            # Extract location data
            location = None
            if 'location' in report_data:
                location = {
                    'lat': report_data['location'].get('latitude'),
                    'lon': report_data['location'].get('longitude'),
                    'address': report_data['location'].get('address'),
                    'accuracy': report_data['location'].get('accuracy')
                }
            
            # Extract media URLs
            media_urls = report_data.get('media_urls', [])
            if 'images' in report_data:
                media_urls.extend(report_data['images'])
            if 'videos' in report_data:
                media_urls.extend(report_data['videos'])
            
            # Create report
            report = RawReport(
                id=report_id,
                source="user_app",
                content=report_data['content'],
                timestamp=timestamp,
                metadata={
                    'user_id': report_data.get('user_id'),
                    'device_info': report_data.get('device_info', {}),
                    'report_type': report_data.get('report_type', 'general'),
                    'severity': report_data.get('severity'),
                    'confidence': report_data.get('confidence'),
                    'verification_status': 'pending'
                },
                location=location,
                media_urls=media_urls if media_urls else None,
                language=report_data.get('language', 'en')
            )
            
            # Add to pending reports
            self.pending_reports.append(report)
            
            print(f"User report submitted: {report_id}")
            return report_id
            
        except Exception as e:
            print(f"Error submitting user report: {e}")
            raise
    
    def submit_bulk_reports(self, reports_data: List[Dict[str, Any]]) -> List[str]:
        """Submit multiple user reports"""
        report_ids = []
        
        for report_data in reports_data:
            try:
                report_id = self.submit_report(report_data)
                report_ids.append(report_id)
            except Exception as e:
                print(f"Error submitting bulk report: {e}")
                report_ids.append(None)
        
        return report_ids
    
    def get_report_status(self, report_id: str) -> Dict[str, Any]:
        """Get status of a submitted report"""
        # Check if report is in pending queue
        for report in self.pending_reports:
            if report.id == report_id:
                return {
                    'id': report_id,
                    'status': 'pending',
                    'submitted_at': report.timestamp,
                    'source': report.source
                }
        
        # Check if report has been processed
        if report_id in self.processed_reports:
            return {
                'id': report_id,
                'status': 'processed',
                'message': 'Report has been processed by the system'
            }
        
        return {
            'id': report_id,
            'status': 'not_found',
            'message': 'Report not found'
        }
    
    def mark_report_processed(self, report_id: str) -> None:
        """Mark a report as processed"""
        self.processed_reports.add(report_id)
    
    async def cleanup(self) -> None:
        """Cleanup user reports source"""
        print("User reports source cleaned up")

class CrowdsourcingPlatform:
    """Platform for managing crowdsourced hazard reports"""
    
    def __init__(self):
        self.user_reports_source = UserReportsSource()
        self.active_reports = {}
        self.user_statistics = {}
    
    async def setup(self) -> None:
        """Setup crowdsourcing platform"""
        await self.user_reports_source.setup()
        print("Crowdsourcing platform setup completed")
    
    def submit_citizen_report(
        self,
        user_id: str,
        content: str,
        location: Optional[Dict[str, Any]] = None,
        media_files: Optional[List[str]] = None,
        report_type: str = "hazard_observation",
        severity: Optional[str] = None,
        language: str = "en"
    ) -> str:
        """Submit a citizen hazard report"""
        
        report_data = {
            'user_id': user_id,
            'content': content,
            'timestamp': datetime.now(),
            'location': location,
            'media_urls': media_files or [],
            'report_type': report_type,
            'severity': severity,
            'language': language,
            'device_info': {
                'platform': 'web',  # or 'mobile'
                'user_agent': 'Ocean Hazard Platform'
            }
        }
        
        report_id = self.user_reports_source.submit_report(report_data)
        
        # Update user statistics
        if user_id not in self.user_statistics:
            self.user_statistics[user_id] = {
                'total_reports': 0,
                'verified_reports': 0,
                'reputation_score': 0.5,
                'last_report_date': None
            }
        
        self.user_statistics[user_id]['total_reports'] += 1
        self.user_statistics[user_id]['last_report_date'] = datetime.now()
        
        # Store active report
        self.active_reports[report_id] = {
            'user_id': user_id,
            'submission_time': datetime.now(),
            'status': 'pending_verification'
        }
        
        return report_id
    
    def verify_report(self, report_id: str, verification_status: str, verifier_id: str) -> bool:
        """Verify a citizen report"""
        if report_id in self.active_reports:
            report_info = self.active_reports[report_id]
            user_id = report_info['user_id']
            
            # Update report status
            report_info['status'] = verification_status
            report_info['verified_by'] = verifier_id
            report_info['verification_time'] = datetime.now()
            
            # Update user reputation
            if verification_status == 'verified':
                self.user_statistics[user_id]['verified_reports'] += 1
                # Increase reputation score
                current_score = self.user_statistics[user_id]['reputation_score']
                self.user_statistics[user_id]['reputation_score'] = min(1.0, current_score + 0.1)
            
            elif verification_status == 'false_alarm':
                # Decrease reputation score
                current_score = self.user_statistics[user_id]['reputation_score']
                self.user_statistics[user_id]['reputation_score'] = max(0.0, current_score - 0.05)
            
            return True
        
        return False
    
    def get_user_reputation(self, user_id: str) -> Dict[str, Any]:
        """Get user reputation and statistics"""
        if user_id in self.user_statistics:
            stats = self.user_statistics[user_id].copy()
            
            # Calculate accuracy rate
            if stats['total_reports'] > 0:
                stats['accuracy_rate'] = stats['verified_reports'] / stats['total_reports']
            else:
                stats['accuracy_rate'] = 0.0
            
            return stats
        
        return {
            'total_reports': 0,
            'verified_reports': 0,
            'reputation_score': 0.5,
            'accuracy_rate': 0.0,
            'last_report_date': None
        }
    
    def get_trending_locations(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get locations with high report activity"""
        # This is a simplified implementation
        # In production, you'd analyze report locations and frequencies
        trending = []
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        # Group reports by location
        location_counts = {}
        for report_id, report_info in self.active_reports.items():
            if report_info['submission_time'] >= cutoff_time:
                # This would need actual location data from reports
                location_key = "example_location"  # Placeholder
                location_counts[location_key] = location_counts.get(location_key, 0) + 1
        
        # Sort by count
        for location, count in sorted(location_counts.items(), key=lambda x: x[1], reverse=True):
            trending.append({
                'location': location,
                'report_count': count,
                'trend_score': count / hours  # Reports per hour
            })
        
        return trending[:10]  # Top 10 trending locations
    
    async def cleanup(self) -> None:
        """Cleanup crowdsourcing platform"""
        await self.user_reports_source.cleanup()
        print("Crowdsourcing platform cleaned up")