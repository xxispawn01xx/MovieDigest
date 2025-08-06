"""
Email outreach system for MovieDigest AI - Entertainment industry marketing
Based on RADflow project patterns with SendGrid integration
"""
import streamlit as st
import pandas as pd
import json
from datetime import datetime, timedelta
from pathlib import Path
import re
from typing import Dict, List, Optional
import time
import os

# Import SendGrid for email sending
try:
    from sendgrid import SendGridAPIClient
    from sendgrid.helpers.mail import Mail, Email, To, Content
    SENDGRID_AVAILABLE = True
except ImportError:
    SENDGRID_AVAILABLE = False

class EmailOutreachSystem:
    """Email marketing system for entertainment industry outreach."""
    
    def __init__(self):
        self.templates_dir = Path("marketing/email_templates")
        self.prospects_dir = Path("marketing/prospect_database")
        self.campaigns_dir = Path("marketing/campaigns")
        
        # Create directories if they don't exist
        for directory in [self.templates_dir, self.prospects_dir, self.campaigns_dir]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def load_email_template(self, template_name: str) -> str:
        """Load email template from file."""
        template_path = self.templates_dir / f"{template_name}.html"
        if template_path.exists():
            return template_path.read_text()
        else:
            st.error(f"Template {template_name} not found")
            return ""
    
    def load_prospects(self, database_file: str) -> pd.DataFrame:
        """Load prospect database from CSV."""
        prospects_path = self.prospects_dir / database_file
        if prospects_path.exists():
            return pd.read_csv(prospects_path)
        else:
            st.error(f"Prospect database {database_file} not found")
            return pd.DataFrame()
    
    def personalize_email(self, template: str, prospect_data: Dict) -> str:
        """Personalize email template with prospect data."""
        personalized = template
        
        # Standard personalizations
        replacements = {
            '{{firstName}}': prospect_data.get('first_name', 'there'),
            '{{lastName}}': prospect_data.get('last_name', ''),
            '{{company}}': prospect_data.get('company', 'your organization'),
            '{{title}}': prospect_data.get('title', 'executive'),
            '{{industry}}': prospect_data.get('industry_segment', 'entertainment'),
            '{{location}}': prospect_data.get('location', ''),
        }
        
        for placeholder, value in replacements.items():
            personalized = personalized.replace(placeholder, str(value))
        
        return personalized
    
    def send_email(self, to_email: str, subject: str, html_content: str, 
                   from_email: str = "hello@moviedigest.ai", from_name: str = "MovieDigest AI") -> bool:
        """Send email using SendGrid."""
        if not SENDGRID_AVAILABLE:
            st.error("SendGrid library not installed. Please install: pip install sendgrid")
            return False
        
        sendgrid_key = os.environ.get('SENDGRID_API_KEY')
        if not sendgrid_key:
            st.error("SENDGRID_API_KEY environment variable not set")
            return False
        
        try:
            sg = SendGridAPIClient(sendgrid_key)
            
            message = Mail(
                from_email=Email(from_email, from_name),
                to_emails=To(to_email),
                subject=subject,
                html_content=Content("text/html", html_content)
            )
            
            response = sg.send(message)
            return response.status_code == 202
            
        except Exception as e:
            st.error(f"SendGrid error: {e}")
            return False
    
    def create_campaign(self, campaign_name: str, template_name: str, 
                       prospect_file: str, subject_line: str, 
                       send_delay_hours: int = 24) -> Dict:
        """Create a new email campaign."""
        
        campaign_data = {
            'name': campaign_name,
            'template': template_name,
            'prospect_file': prospect_file,
            'subject_line': subject_line,
            'send_delay_hours': send_delay_hours,
            'created_at': datetime.now().isoformat(),
            'status': 'draft',
            'emails_sent': 0,
            'responses': 0,
            'opens': 0,  # Would need tracking pixels for this
            'clicks': 0   # Would need link tracking for this
        }
        
        # Save campaign configuration
        campaign_file = self.campaigns_dir / f"{campaign_name.replace(' ', '_').lower()}.json"
        campaign_file.write_text(json.dumps(campaign_data, indent=2))
        
        return campaign_data
    
    def get_campaigns(self) -> List[Dict]:
        """Get all saved campaigns."""
        campaigns = []
        for campaign_file in self.campaigns_dir.glob("*.json"):
            try:
                campaign_data = json.loads(campaign_file.read_text())
                campaigns.append(campaign_data)
            except Exception as e:
                st.error(f"Error loading campaign {campaign_file.name}: {e}")
        
        return sorted(campaigns, key=lambda x: x.get('created_at', ''), reverse=True)
    
    def preview_campaign(self, template_name: str, prospect_file: str, num_preview: int = 3) -> List[str]:
        """Preview personalized emails for campaign."""
        template = self.load_email_template(template_name)
        prospects = self.load_prospects(prospect_file)
        
        previews = []
        for _, prospect in prospects.head(num_preview).iterrows():
            personalized = self.personalize_email(template, prospect.to_dict())
            previews.append(personalized)
        
        return previews
    
    def execute_campaign(self, campaign_name: str, dry_run: bool = True) -> Dict:
        """Execute email campaign."""
        campaign_file = self.campaigns_dir / f"{campaign_name.replace(' ', '_').lower()}.json"
        
        if not campaign_file.exists():
            return {"error": "Campaign not found"}
        
        campaign_data = json.loads(campaign_file.read_text())
        
        # Load template and prospects
        template = self.load_email_template(campaign_data['template'])
        prospects = self.load_prospects(campaign_data['prospect_file'])
        
        results = {
            'attempted': 0,
            'sent': 0,
            'failed': 0,
            'errors': []
        }
        
        for _, prospect in prospects.iterrows():
            results['attempted'] += 1
            
            # Personalize email
            personalized_email = self.personalize_email(template, prospect.to_dict())
            
            if dry_run:
                # Just simulate sending
                results['sent'] += 1
                time.sleep(0.1)  # Small delay to simulate processing
            else:
                # Actually send the email
                success = self.send_email(
                    to_email=prospect['email'],
                    subject=campaign_data['subject_line'],
                    html_content=personalized_email
                )
                
                if success:
                    results['sent'] += 1
                else:
                    results['failed'] += 1
                    results['errors'].append(f"Failed to send to {prospect['email']}")
                
                # Delay between emails to avoid rate limiting
                time.sleep(campaign_data.get('send_delay_hours', 24) * 0.001)  # Convert to seconds for demo
        
        # Update campaign statistics
        campaign_data['emails_sent'] = results['sent']
        campaign_data['status'] = 'completed' if results['failed'] == 0 else 'partial'
        campaign_data['last_run'] = datetime.now().isoformat()
        
        # Save updated campaign data
        campaign_file.write_text(json.dumps(campaign_data, indent=2))
        
        return results

def show_email_outreach_page():
    """Main email outreach management page."""
    
    st.header("📧 Email Outreach System")
    st.write("Marketing automation for MovieDigest AI - Entertainment industry outreach")
    
    # Initialize system
    outreach_system = EmailOutreachSystem()
    
    # Tabs for different functions
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📝 Create Campaign", 
        "👁️ Preview Emails", 
        "🚀 Launch Campaign",
        "📊 Campaign Analytics",
        "⚙️ Settings"
    ])
    
    with tab1:
        st.subheader("Create New Email Campaign")
        
        col1, col2 = st.columns(2)
        
        with col1:
            campaign_name = st.text_input(
                "Campaign Name",
                placeholder="MovieDigest Q1 2025 Outreach"
            )
            
            subject_line = st.text_input(
                "Email Subject Line",
                value="Reduce video analysis time by 75% with MovieDigest AI",
                help="Personalization variables: {{firstName}}, {{company}}, {{title}}"
            )
        
        with col2:
            # Template selection
            available_templates = ["entertainment_industry_template"]  # Could scan directory
            template_name = st.selectbox(
                "Email Template",
                available_templates,
                help="HTML email templates for different industries"
            )
            
            # Prospect database selection
            available_databases = ["entertainment_contacts.csv"]  # Could scan directory
            prospect_file = st.selectbox(
                "Prospect Database",
                available_databases,
                help="CSV files with prospect contact information"
            )
        
        # Advanced settings
        with st.expander("🔧 Advanced Campaign Settings"):
            col1, col2 = st.columns(2)
            
            with col1:
                send_delay = st.slider(
                    "Delay between emails (hours)",
                    1, 72, 24,
                    help="Rate limiting to avoid spam detection"
                )
                
                from_email = st.text_input(
                    "From Email Address",
                    value="hello@moviedigest.ai"
                )
            
            with col2:
                from_name = st.text_input(
                    "From Name",
                    value="MovieDigest AI Team"
                )
                
                track_opens = st.checkbox("Track Email Opens", value=True)
                track_clicks = st.checkbox("Track Link Clicks", value=True)
        
        if st.button("💾 Create Campaign", type="primary"):
            if campaign_name and subject_line:
                campaign_data = outreach_system.create_campaign(
                    campaign_name=campaign_name,
                    template_name=template_name,
                    prospect_file=prospect_file,
                    subject_line=subject_line,
                    send_delay_hours=send_delay
                )
                
                st.success(f"✅ Campaign '{campaign_name}' created successfully!")
                st.json(campaign_data)
            else:
                st.error("Please fill in campaign name and subject line")
    
    with tab2:
        st.subheader("📧 Preview Campaign Emails")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Campaign selection
            campaigns = outreach_system.get_campaigns()
            if campaigns:
                campaign_names = [c['name'] for c in campaigns]
                selected_campaign = st.selectbox("Select Campaign", campaign_names)
                
                num_previews = st.slider("Number of previews", 1, 10, 3)
                
                if st.button("🔍 Generate Previews"):
                    campaign_data = next(c for c in campaigns if c['name'] == selected_campaign)
                    
                    previews = outreach_system.preview_campaign(
                        template_name=campaign_data['template'],
                        prospect_file=campaign_data['prospect_file'],
                        num_preview=num_previews
                    )
                    
                    st.session_state.email_previews = previews
            else:
                st.info("No campaigns found. Create a campaign first.")
        
        with col2:
            if hasattr(st.session_state, 'email_previews'):
                for i, preview in enumerate(st.session_state.email_previews):
                    st.write(f"**Preview {i+1}:**")
                    st.components.v1.html(preview, height=400, scrolling=True)
                    st.divider()
    
    with tab3:
        st.subheader("🚀 Launch Email Campaign")
        
        # Campaign selection
        campaigns = outreach_system.get_campaigns()
        if campaigns:
            campaign_names = [c['name'] for c in campaigns]
            selected_campaign = st.selectbox("Select Campaign to Launch", campaign_names)
            
            campaign_data = next(c for c in campaigns if c['name'] == selected_campaign)
            
            # Show campaign details
            st.write("**Campaign Details:**")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Template", campaign_data['template'])
                st.metric("Subject", campaign_data['subject_line'][:30] + "...")
            
            with col2:
                prospects = outreach_system.load_prospects(campaign_data['prospect_file'])
                st.metric("Total Prospects", len(prospects))
                st.metric("Status", campaign_data['status'])
            
            with col3:
                st.metric("Emails Sent", campaign_data.get('emails_sent', 0))
                st.metric("Delay (hours)", campaign_data.get('send_delay_hours', 24))
            
            # Launch options
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🧪 Test Run (Dry Run)", type="secondary"):
                    with st.spinner("Running test campaign..."):
                        results = outreach_system.execute_campaign(selected_campaign, dry_run=True)
                    
                    st.success("Test completed!")
                    st.json(results)
            
            with col2:
                st.warning("⚠️ Live Campaign - Emails will be sent!")
                
                # Check for SendGrid API key
                if not os.environ.get('SENDGRID_API_KEY'):
                    st.error("SENDGRID_API_KEY not configured. Cannot send emails.")
                    if st.button("Configure SendGrid API Key"):
                        st.info("Go to Settings tab to configure SendGrid API key")
                else:
                    if st.button("🚀 Launch Live Campaign", type="primary"):
                        confirm = st.checkbox("I confirm this will send real emails")
                        
                        if confirm:
                            with st.spinner("Sending emails..."):
                                results = outreach_system.execute_campaign(selected_campaign, dry_run=False)
                            
                            if results.get('error'):
                                st.error(results['error'])
                            else:
                                st.success(f"Campaign launched! Sent {results['sent']} emails.")
                                st.json(results)
        else:
            st.info("No campaigns found. Create a campaign first.")
    
    with tab4:
        st.subheader("📊 Campaign Analytics")
        
        campaigns = outreach_system.get_campaigns()
        if campaigns:
            # Campaign performance overview
            st.write("**Campaign Performance Overview:**")
            
            campaign_df = pd.DataFrame(campaigns)
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_campaigns = len(campaigns)
                st.metric("Total Campaigns", total_campaigns)
            
            with col2:
                total_emails = campaign_df['emails_sent'].sum()
                st.metric("Total Emails Sent", int(total_emails))
            
            with col3:
                active_campaigns = len(campaign_df[campaign_df['status'] == 'active'])
                st.metric("Active Campaigns", active_campaigns)
            
            with col4:
                completed_campaigns = len(campaign_df[campaign_df['status'] == 'completed'])
                st.metric("Completed Campaigns", completed_campaigns)
            
            # Campaign details table
            st.write("**Campaign Details:**")
            display_columns = ['name', 'status', 'emails_sent', 'created_at']
            st.dataframe(
                campaign_df[display_columns].round(2),
                use_container_width=True
            )
            
            # Individual campaign analysis
            st.write("**Individual Campaign Analysis:**")
            selected_campaign = st.selectbox("Select Campaign for Details", [c['name'] for c in campaigns])
            
            campaign_data = next(c for c in campaigns if c['name'] == selected_campaign)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.json(campaign_data)
            
            with col2:
                # Load prospect data for analysis
                prospects = outreach_system.load_prospects(campaign_data['prospect_file'])
                
                st.write("**Prospect Breakdown:**")
                
                # Industry segment analysis
                if 'industry_segment' in prospects.columns:
                    segment_counts = prospects['industry_segment'].value_counts()
                    st.bar_chart(segment_counts)
                
                # Authority score distribution
                if 'authority_score' in prospects.columns:
                    st.write("**Authority Score Distribution:**")
                    st.histogram_chart(prospects['authority_score'])
        else:
            st.info("No campaign data available.")
    
    with tab5:
        st.subheader("⚙️ Email System Settings")
        
        # SendGrid configuration
        st.write("**SendGrid Configuration:**")
        
        current_key = os.environ.get('SENDGRID_API_KEY', '')
        masked_key = f"{current_key[:8]}..." if current_key else "Not configured"
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"Current API Key: {masked_key}")
            
            if not current_key:
                st.warning("⚠️ SendGrid API key not configured")
                st.write("To configure:")
                st.code("1. Get API key from SendGrid dashboard\n2. Add to Replit Secrets as SENDGRID_API_KEY")
        
        with col2:
            st.write("**Email Settings:**")
            
            default_from_email = st.text_input(
                "Default From Email",
                value=st.session_state.get('default_from_email', 'hello@moviedigest.ai')
            )
            
            default_from_name = st.text_input(
                "Default From Name", 
                value=st.session_state.get('default_from_name', 'MovieDigest AI')
            )
            
            if st.button("💾 Save Email Settings"):
                st.session_state.default_from_email = default_from_email
                st.session_state.default_from_name = default_from_name
                st.success("Settings saved!")
        
        # Template management
        st.write("**Template Management:**")
        
        uploaded_template = st.file_uploader(
            "Upload New Email Template",
            type=['html'],
            help="Upload HTML email template files"
        )
        
        if uploaded_template:
            template_name = st.text_input(
                "Template Name",
                value=uploaded_template.name.replace('.html', '')
            )
            
            if st.button("📤 Upload Template"):
                template_path = outreach_system.templates_dir / f"{template_name}.html"
                template_path.write_bytes(uploaded_template.read())
                st.success(f"Template '{template_name}' uploaded successfully!")
        
        # Prospect database management
        st.write("**Prospect Database Management:**")
        
        uploaded_prospects = st.file_uploader(
            "Upload Prospect Database",
            type=['csv'],
            help="Upload CSV files with prospect contact information"
        )
        
        if uploaded_prospects:
            database_name = st.text_input(
                "Database Name",
                value=uploaded_prospects.name
            )
            
            if st.button("📤 Upload Prospect Database"):
                database_path = outreach_system.prospects_dir / database_name
                database_path.write_bytes(uploaded_prospects.read())
                st.success(f"Database '{database_name}' uploaded successfully!")
        
        # System status
        st.write("**System Status:**")
        
        status_checks = {
            "SendGrid Available": SENDGRID_AVAILABLE,
            "API Key Configured": bool(current_key),
            "Templates Directory": outreach_system.templates_dir.exists(),
            "Prospects Directory": outreach_system.prospects_dir.exists(),
            "Campaigns Directory": outreach_system.campaigns_dir.exists(),
        }
        
        for check, status in status_checks.items():
            if status:
                st.success(f"✅ {check}")
            else:
                st.error(f"❌ {check}")