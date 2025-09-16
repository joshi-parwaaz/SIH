"""Alert generation and notification system."""

import smtplib
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
import logging

# Communication and notifications
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

logger = logging.getLogger(__name__)

class AlertGenerator:
    """Generate and distribute alerts for ocean hazard predictions."""
    
    def __init__(self, config: Dict = None):
        self.config = config or self._load_default_config()
        self.alert_history = {}
        self.active_alerts = {}
        self.notification_channels = {}
        
    def _load_default_config(self) -> Dict:
        """Load default alert configuration."""
        
        return {
            'alert_thresholds': {
                'Critical': {'min_risk_score': 0.9, 'immediate': True},
                'High': {'min_risk_score': 0.7, 'immediate': False},
                'Medium': {'min_risk_score': 0.5, 'immediate': False},
                'Low': {'min_risk_score': 0.3, 'immediate': False}
            },
            'notification_settings': {
                'email': {
                    'enabled': False,
                    'smtp_server': '',
                    'smtp_port': 587,
                    'username': '',
                    'password': '',
                    'from_address': '',
                    'recipient_groups': {
                        'emergency': [],
                        'monitoring': [],
                        'public': []
                    }
                },
                'webhook': {
                    'enabled': False,
                    'urls': []
                },
                'sms': {
                    'enabled': False,
                    'api_key': '',
                    'service_url': ''
                }
            },
            'alert_templates': {
                'Critical': {
                    'subject': 'CRITICAL OCEAN HAZARD ALERT - {hazard_type}',
                    'priority': 'URGENT',
                    'color': '#FF0000'
                },
                'High': {
                    'subject': 'HIGH RISK OCEAN HAZARD ALERT - {hazard_type}',
                    'priority': 'HIGH',
                    'color': '#FF6600'
                },
                'Medium': {
                    'subject': 'Ocean Hazard Advisory - {hazard_type}',
                    'priority': 'MEDIUM',
                    'color': '#FF9900'
                },
                'Low': {
                    'subject': 'Ocean Hazard Update - {hazard_type}',
                    'priority': 'LOW',
                    'color': '#FFCC00'
                }
            },
            'cooldown_periods': {
                'Critical': 1,    # hours
                'High': 3,
                'Medium': 6,
                'Low': 12
            }
        }
    
    def generate_alert(self, risk_assessment: Dict, 
                      alert_type: str = 'automatic') -> Dict:
        """Generate alert based on risk assessment."""
        
        risk_score = risk_assessment.get('risk_score', 0)
        risk_level = risk_assessment.get('risk_level', 'Low')
        hazard_type = risk_assessment.get('hazard_type', 'unknown')
        location = risk_assessment.get('location', {})
        
        logger.info(f"Generating {risk_level} alert for {hazard_type} at {location}")
        
        # Check if alert meets threshold
        threshold_config = self.config['alert_thresholds'].get(risk_level, {})
        min_risk_score = threshold_config.get('min_risk_score', 1.0)
        
        if risk_score < min_risk_score:
            logger.info(f"Risk score {risk_score} below threshold {min_risk_score}. No alert generated.")
            return {'alert_generated': False, 'reason': 'Below threshold'}
        
        # Check cooldown period
        if self._is_in_cooldown(location, hazard_type, risk_level):
            logger.info("Alert in cooldown period. Skipping alert generation.")
            return {'alert_generated': False, 'reason': 'Cooldown period'}
        
        # Generate alert ID
        alert_id = f"{hazard_type}_{int(datetime.now().timestamp())}"
        
        # Create alert object
        alert = {
            'alert_id': alert_id,
            'generated_at': datetime.now().isoformat(),
            'alert_type': alert_type,
            'risk_level': risk_level,
            'hazard_type': hazard_type,
            'location': location,
            'risk_assessment': risk_assessment,
            'alert_details': self._create_alert_details(risk_assessment),
            'notification_status': {},
            'is_active': True
        }
        
        # Store alert
        self.active_alerts[alert_id] = alert
        
        # Send notifications
        notification_results = self._send_notifications(alert)
        alert['notification_status'] = notification_results
        
        # Store in history
        if alert_id not in self.alert_history:
            self.alert_history[alert_id] = []
        self.alert_history[alert_id].append(alert)
        
        logger.info(f"Alert {alert_id} generated and distributed")
        
        return {
            'alert_generated': True,
            'alert_id': alert_id,
            'alert': alert,
            'notifications_sent': sum(1 for result in notification_results.values() if result.get('success', False))
        }
    
    def _create_alert_details(self, risk_assessment: Dict) -> Dict:
        """Create detailed alert information."""
        
        risk_level = risk_assessment.get('risk_level', 'Unknown')
        hazard_type = risk_assessment.get('hazard_type', 'unknown')
        location = risk_assessment.get('location', {})
        risk_score = risk_assessment.get('risk_score', 0)
        confidence = risk_assessment.get('confidence_score', 0)
        recommendations = risk_assessment.get('recommendations', [])
        
        # Get template configuration
        template = self.config['alert_templates'].get(risk_level, {})
        
        # Create alert message
        message_parts = []
        
        # Header
        if risk_level == 'Critical':
            message_parts.append("⚠️ CRITICAL OCEAN HAZARD ALERT ⚠️")
        elif risk_level == 'High':
            message_parts.append("🔴 HIGH RISK OCEAN HAZARD ALERT")
        elif risk_level == 'Medium':
            message_parts.append("🟡 Ocean Hazard Advisory")
        else:
            message_parts.append("ℹ️ Ocean Hazard Update")
        
        message_parts.append("")
        
        # Details
        message_parts.extend([
            f"Hazard Type: {hazard_type.title()}",
            f"Risk Level: {risk_level}",
            f"Risk Score: {risk_score:.3f}/1.000",
            f"Confidence: {confidence:.3f}/1.000",
            ""
        ])
        
        # Location
        if location:
            lat = location.get('latitude', 'Unknown')
            lon = location.get('longitude', 'Unknown')
            message_parts.extend([
                "Location:",
                f"  Latitude: {lat}",
                f"  Longitude: {lon}",
                ""
            ])
        
        # Time information
        time_window = risk_assessment.get('time_window', 'Unknown')
        message_parts.extend([
            f"Time Frame: {time_window}",
            f"Assessment Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}",
            ""
        ])
        
        # Recommendations
        if recommendations:
            message_parts.append("Recommended Actions:")
            for i, rec in enumerate(recommendations[:5], 1):  # Limit to top 5
                message_parts.append(f"  {i}. {rec}")
            message_parts.append("")
        
        # Footer
        if risk_level in ['Critical', 'High']:
            message_parts.append("Please take immediate action and monitor official channels for updates.")
        else:
            message_parts.append("Continue monitoring conditions and stay prepared.")
        
        alert_details = {
            'subject': template.get('subject', 'Ocean Hazard Alert').format(hazard_type=hazard_type.title()),
            'message': "\n".join(message_parts),
            'priority': template.get('priority', 'MEDIUM'),
            'color': template.get('color', '#FFCC00'),
            'immediate': self.config['alert_thresholds'].get(risk_level, {}).get('immediate', False),
            'expiry_time': (datetime.now() + timedelta(hours=24)).isoformat()
        }
        
        return alert_details
    
    def _is_in_cooldown(self, location: Dict, hazard_type: str, 
                       risk_level: str) -> bool:
        """Check if location/hazard combination is in cooldown period."""
        
        if not location:
            return False
        
        cooldown_hours = self.config['cooldown_periods'].get(risk_level, 24)
        cooldown_delta = timedelta(hours=cooldown_hours)
        current_time = datetime.now()
        
        # Check recent alerts for same location and hazard type
        for alert_data in self.alert_history.values():
            for alert in alert_data:
                if not alert.get('is_active', False):
                    continue
                
                alert_time = datetime.fromisoformat(alert['generated_at'].replace('Z', '+00:00'))
                if current_time - alert_time.replace(tzinfo=None) < cooldown_delta:
                    # Check if same location (within ~10km)
                    alert_location = alert.get('location', {})
                    if (abs(location.get('latitude', 0) - alert_location.get('latitude', 999)) < 0.1 and
                        abs(location.get('longitude', 0) - alert_location.get('longitude', 999)) < 0.1 and
                        alert.get('hazard_type') == hazard_type):
                        return True
        
        return False
    
    def _send_notifications(self, alert: Dict) -> Dict:
        """Send notifications through configured channels."""
        
        notification_results = {}
        risk_level = alert['risk_level']
        
        # Determine recipient groups based on risk level
        if risk_level == 'Critical':
            recipient_groups = ['emergency', 'monitoring', 'public']
        elif risk_level == 'High':
            recipient_groups = ['emergency', 'monitoring']
        else:
            recipient_groups = ['monitoring']
        
        # Send email notifications
        if self.config['notification_settings']['email']['enabled']:
            email_result = self._send_email_notification(alert, recipient_groups)
            notification_results['email'] = email_result
        
        # Send webhook notifications
        if self.config['notification_settings']['webhook']['enabled']:
            webhook_result = self._send_webhook_notification(alert)
            notification_results['webhook'] = webhook_result
        
        # Send SMS notifications (for critical alerts)
        if (risk_level == 'Critical' and 
            self.config['notification_settings']['sms']['enabled']):
            sms_result = self._send_sms_notification(alert, recipient_groups)
            notification_results['sms'] = sms_result
        
        return notification_results
    
    def _send_email_notification(self, alert: Dict, 
                                recipient_groups: List[str]) -> Dict:
        """Send email notification."""
        
        email_config = self.config['notification_settings']['email']
        
        if not email_config.get('enabled', False):
            return {'success': False, 'reason': 'Email not enabled'}
        
        try:
            # Collect recipients
            recipients = []
            for group in recipient_groups:
                recipients.extend(email_config['recipient_groups'].get(group, []))
            
            if not recipients:
                return {'success': False, 'reason': 'No recipients configured'}
            
            # Create email message
            msg = MimeMultipart()
            msg['From'] = email_config['from_address']
            msg['To'] = ', '.join(recipients)
            msg['Subject'] = alert['alert_details']['subject']
            
            # Email body
            body = alert['alert_details']['message']
            msg.attach(MimeText(body, 'plain'))
            
            logger.info(f"Email notification prepared for {len(recipients)} recipients")
            
            return {
                'success': True,
                'recipients_count': len(recipients),
                'sent_at': datetime.now().isoformat(),
                'note': 'Email sending simulated - SMTP not configured'
            }
            
        except Exception as e:
            logger.error(f"Error preparing email notification: {e}")
            return {'success': False, 'error': str(e)}
    
    def _send_webhook_notification(self, alert: Dict) -> Dict:
        """Send webhook notification."""
        
        if not HAS_REQUESTS:
            return {'success': False, 'reason': 'Requests library not available'}
        
        webhook_config = self.config['notification_settings']['webhook']
        
        if not webhook_config.get('enabled', False):
            return {'success': False, 'reason': 'Webhook not enabled'}
        
        webhook_urls = webhook_config.get('urls', [])
        if not webhook_urls:
            return {'success': False, 'reason': 'No webhook URLs configured'}
        
        # Prepare webhook payload
        payload = {
            'alert_id': alert['alert_id'],
            'timestamp': alert['generated_at'],
            'risk_level': alert['risk_level'],
            'hazard_type': alert['hazard_type'],
            'location': alert['location'],
            'risk_score': alert['risk_assessment'].get('risk_score', 0),
            'message': alert['alert_details']['message'],
            'priority': alert['alert_details']['priority']
        }
        
        logger.info(f"Webhook notification prepared for {len(webhook_urls)} URLs")
        
        return {
            'success': True,
            'total_webhooks': len(webhook_urls),
            'payload': payload,
            'note': 'Webhook sending simulated'
        }
    
    def _send_sms_notification(self, alert: Dict, 
                              recipient_groups: List[str]) -> Dict:
        """Send SMS notification (simplified implementation)."""
        
        logger.info(f"SMS notification prepared for alert {alert['alert_id']}")
        
        return {
            'success': True,
            'reason': 'SMS simulation - integration required',
            'recipients_notified': recipient_groups
        }
    
    def export_alert_data(self, output_file: str = None) -> str:
        """Export alert data for analysis."""
        
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"alert_data_{timestamp}.json"
        
        export_data = {
            'export_metadata': {
                'exported_at': datetime.now().isoformat(),
                'export_type': 'alert_data_backup'
            },
            'active_alerts': self.active_alerts,
            'alert_history': self.alert_history,
            'config': self.config
        }
        
        try:
            with open(output_file, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            logger.info(f"Alert data exported to {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"Error exporting alert data: {e}")
            return ""