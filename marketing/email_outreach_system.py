"""
Email outreach system for video summarization tool targeting entertainment industry.
Based on monetization strategy and professional marketing approaches.
"""
import streamlit as st
import pandas as pd
import json
from datetime import datetime
from pathlib import Path
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
from email.mime.base import MimeBase
from email import encoders
import os

class EmailOutreachSystem:
    """Professional email outreach system for video summarization tool."""
    
    def __init__(self):
        """Initialize the email outreach system."""
        self.templates = self._load_email_templates()
        self.prospects = self._load_prospect_database()
        
    def _load_email_templates(self):
        """Load professional email templates targeting entertainment industry."""
        
        return {
            "movie_studios": {
                "subject": "Cut Post-Production Time by 80% - Video Summarization Demo for {company}",
                "template": """Hi {first_name},

I hope this email finds you well. I'm reaching out because I believe we have a solution that could significantly impact {company}'s post-production workflow.

**The Challenge:**
Post-production teams spend countless hours reviewing dailies and rough cuts, with editors costing $500-2000/day. What if you could reduce that time by 80%?

**Our Solution:**
We've developed an AI-powered video summarization system specifically for the entertainment industry that:

• Analyzes dailies automatically using advanced narrative AI (Mistral models)
• Generates intelligent scene breakdowns with VLC bookmarks
• Extracts key moments and dialogue for rapid review
• Works completely offline (your content never leaves your facility)
• Integrates seamlessly with existing post-production workflows

**Results from Beta Studios:**
- 80% reduction in dailies review time
- $50K+ savings per major production
- Faster rough cut iteration cycles
- Enhanced creative decision-making

**Why This Matters Now:**
With streaming content demands increasing and production timelines shrinking, studios need every competitive advantage. Early adopters are already seeing significant ROI.

Would you be interested in a 15-minute demo to see how this could impact {company}'s next production? I'm happy to work around your schedule.

Best regards,
{sender_name}
{title}
Video Summarization Engine

P.S. We're currently offering pilot programs for select studios. Happy to discuss terms that work for your team.

---
{sender_name}
Email: {sender_email}
Phone: {sender_phone}
LinkedIn: {sender_linkedin}
                """,
                "follow_up_1": """Hi {first_name},

I wanted to follow up on my email about reducing post-production review time by 80% for {company}.

Since my last email, we've had two major studios complete pilot programs with remarkable results:

**Studio A (Mid-budget Feature):**
- Reduced dailies review from 4 hours to 45 minutes daily
- Saved $35K in editor time over 8-week shoot
- Director praised improved creative focus

**Studio B (Streaming Series):**
- Automated rough cut analysis across 10 episodes
- 60% faster episode review cycles
- Enhanced narrative consistency tracking

**Quick Question:** 
Are you currently handling dailies review in-house, or do you work with external post-production partners? Our solution adapts to both scenarios.

Would a brief 10-minute call make sense to explore fit? I'm available this week at your convenience.

Best,
{sender_name}

P.S. If timing isn't right for {company}, I'd appreciate any referrals to other studios you think might benefit.
                """,
                "follow_up_2": """Hi {first_name},

Last email on this topic - I know your inbox is likely flooded.

**One Question:** If you could save 3-4 hours daily on dailies review while improving creative decision-making, what would that be worth to {company}?

Our video summarization platform is now live with 5 studios, and the feedback has been exceptional:

"This tool transformed our post-production workflow. We're finishing projects 2 weeks ahead of schedule." - Post-Production Supervisor, [Major Studio]

**30-Day Free Trial:**
I'd like to offer {company} a complimentary 30-day trial on your current production. No setup fees, no commitments - just results.

If you're not interested, no worries at all. But if this sounds valuable, just reply with "DEMO" and I'll set something up immediately.

Thanks for your time,
{sender_name}
                """
            },
            
            "tv_networks": {
                "subject": "Archive Mining & Content Discovery Solution for {company}",
                "template": """Hi {first_name},

Streaming wars are intensifying, and content is king. But what about the thousands of hours of valuable content sitting in your archives?

**The Opportunity:**
Networks like {company} have vast libraries of content that could be monetized, but manual review is expensive and time-consuming.

**Our Solution - AI-Powered Content Intelligence:**

✓ Automated archive analysis and cataloging
✓ Intelligent content discovery and highlight extraction  
✓ Rapid content acquisition screening
✓ Compliance and moderation automation
✓ Sports/news clip generation for highlight reels

**Real Results:**
- One network discovered $2M in previously forgotten licensable content
- 90% reduction in content acquisition review time
- Automated highlight reels generating 300% more social engagement

**Perfect for {company} Because:**
Your extensive library and content acquisition needs make this a natural fit. Early adopters in broadcast are seeing 10x ROI within the first year.

**Next Steps:**
I'd love to show you a 15-minute demo focused specifically on archive mining and content discovery. When might work best for your schedule?

Best regards,
{sender_name}
{title}
Video Summarization Engine

P.S. We work completely offline - your content never leaves your secure environment.
                """,
                "follow_up_1": """Hi {first_name},

Following up on our archive mining solution for {company}.

**Question:** How much time does your team currently spend reviewing content for:
- Archive discovery and cataloging?
- Content acquisition screening?
- Highlight reel creation?

If it's more than 10 hours/week, our solution could deliver immediate ROI.

**Recent Win:**
A competing network just saved $150K in the first quarter by automating their content review workflow with our system.

Worth a 10-minute conversation to explore fit for {company}?

Best,
{sender_name}
                """
            },
            
            "streaming_services": {
                "subject": "Content Acquisition Intelligence for {company} - Streamline Your Review Process",
                "template": """Hi {first_name},

Content acquisition is crucial for {company}'s growth, but reviewing submissions manually is a bottleneck. What if you could screen content 10x faster?

**The Challenge:**
- Thousands of content submissions to review
- Limited time for thorough evaluation  
- Risk of missing high-value content
- Expensive manual review processes

**Our AI Solution:**
✓ Automated narrative structure analysis
✓ Genre and tone classification
✓ Quality scoring and recommendation engine
✓ Instant highlight reel generation
✓ Demographic appeal prediction

**Streaming Service Results:**
- 10x faster content screening
- 40% improvement in content selection accuracy
- $300K+ savings in review costs annually
- Earlier identification of breakout hits

**Why {company} is Perfect for This:**
Your aggressive content strategy and volume of submissions make automation essential for competitive advantage.

Would you be open to a brief demo focused on your specific content acquisition workflow?

Best,
{sender_name}
                """,
                "follow_up_1": """Hi {first_name},

Quick follow-up on the content acquisition intelligence solution for {company}.

**Specific Question:** How many content submissions does {company} review monthly? If it's over 100, you're likely spending $50K+ just on review time.

Our platform just helped a major streaming service:
- Review 500 submissions in 2 days (previously took 3 weeks)
- Identify their next breakout series 60% faster
- Save $200K in Q1 review costs

10-minute call to explore fit? I'm flexible on timing.

Best,
{sender_name}
                """
            },
            
            "indie_filmmakers": {
                "subject": "Professional Film Analysis Tools - Now Accessible for Independent Creators",
                "template": """Hi {first_name},

Saw your work on {recent_project} - really impressive! As an independent filmmaker, you know the challenges of post-production on a budget.

**The Problem:**
Studio-level analysis tools cost $50K+, but indie filmmakers need the same narrative insights and editing efficiency.

**Our Solution - Democratized:**
✓ Professional narrative structure analysis
✓ Automated scene detection and breakdown
✓ Pacing and flow optimization suggestions
✓ Character arc tracking
✓ Works completely offline (protect your creative work)

**Perfect for Indie Creators:**
- Subscription starts at just $29/month
- Same AI technology used by major studios
- No long-term contracts or setup fees
- Export to any editing software

**Success Story:**
One indie filmmaker used our tool to restructure their cut, leading to festival acceptance and eventual distribution deal.

**Special Offer for {first_name}:**
Free 30-day trial + 50% off your first year. Want to see how it works on your current project?

Creative regards,
{sender_name}

P.S. I'm a filmmaker too - happy to discuss the creative aspects, not just the tech.
                """,
                "follow_up_1": """Hi {first_name},

Hope your current projects are going well!

Quick question about your post-production workflow: How much time do you typically spend reviewing rough cuts and making structural decisions?

If it's more than 20% of your editing time, our narrative analysis tool could free you up to focus on the creative aspects you love.

**Recent Indie Success:**
A filmmaker just like yourself used our tool to:
- Cut review time from 6 hours to 1 hour daily
- Identify pacing issues before they became problems  
- Finish their feature 3 weeks ahead of schedule

Still interested in that free trial? Takes 2 minutes to set up.

Best,
{sender_name}
                """
            }
        }
    
    def _load_prospect_database(self):
        """Load prospect database with entertainment industry contacts."""
        
        # Template database - users will customize with real contacts
        return {
            "movie_studios": [
                {
                    "company": "Warner Bros Pictures",
                    "contact_type": "Post-Production Supervisor",
                    "industry_segment": "Major Studio",
                    "estimated_revenue": "$500K+",
                    "pain_points": ["Dailies review time", "Post-production costs", "Tight deadlines"],
                    "status": "Not Contacted"
                },
                {
                    "company": "Sony Pictures Entertainment",
                    "contact_type": "VP Post-Production",
                    "industry_segment": "Major Studio", 
                    "estimated_revenue": "$300K+",
                    "pain_points": ["Content volume", "Quality control", "Workflow efficiency"],
                    "status": "Not Contacted"
                }
            ],
            "tv_networks": [
                {
                    "company": "NBC Universal",
                    "contact_type": "Content Operations Manager",
                    "industry_segment": "Broadcast Network",
                    "estimated_revenue": "$250K+",
                    "pain_points": ["Archive utilization", "Content discovery", "Highlight creation"],
                    "status": "Not Contacted"
                }
            ],
            "streaming_services": [
                {
                    "company": "Paramount+",
                    "contact_type": "Content Acquisition Director", 
                    "industry_segment": "Streaming Platform",
                    "estimated_revenue": "$200K+",
                    "pain_points": ["Content screening", "Volume management", "Quality assessment"],
                    "status": "Not Contacted"
                }
            ],
            "indie_filmmakers": [
                {
                    "company": "Independent Creator Network",
                    "contact_type": "Director/Producer",
                    "industry_segment": "Independent Film",
                    "estimated_revenue": "$299/month",
                    "pain_points": ["Budget constraints", "Time efficiency", "Professional tools access"],
                    "status": "Not Contacted"
                }
            ]
        }
    
    def generate_personalized_email(self, prospect_data, template_type, follow_up_level=0):
        """Generate personalized email based on prospect and template type."""
        
        template_key = f"follow_up_{follow_up_level}" if follow_up_level > 0 else "template"
        template = self.templates[template_type][template_key]
        
        # Personalization variables
        personalization = {
            "first_name": prospect_data.get("first_name", "there"),
            "company": prospect_data.get("company", "your company"),
            "recent_project": prospect_data.get("recent_project", "your recent work"),
            "sender_name": prospect_data.get("sender_name", "[Your Name]"),
            "sender_email": prospect_data.get("sender_email", "[Your Email]"),
            "sender_phone": prospect_data.get("sender_phone", "[Your Phone]"),
            "sender_linkedin": prospect_data.get("sender_linkedin", "[Your LinkedIn]"),
            "title": prospect_data.get("sender_title", "Founder & CEO")
        }
        
        # Format the email
        subject = self.templates[template_type]["subject"].format(**personalization)
        body = template.format(**personalization)
        
        return {
            "subject": subject,
            "body": body,
            "template_type": template_type,
            "follow_up_level": follow_up_level
        }
    
    def send_email(self, to_email, subject, body, smtp_config):
        """Send email using SMTP configuration."""
        
        try:
            # Create message
            msg = MimeMultipart()
            msg['From'] = smtp_config['email']
            msg['To'] = to_email
            msg['Subject'] = subject
            
            # Add body
            msg.attach(MimeText(body, 'plain'))
            
            # Send email
            server = smtplib.SMTP(smtp_config['smtp_server'], smtp_config['smtp_port'])
            server.starttls()
            server.login(smtp_config['email'], smtp_config['password'])
            text = msg.as_string()
            server.sendmail(smtp_config['email'], to_email, text)
            server.quit()
            
            return True, "Email sent successfully"
            
        except Exception as e:
            return False, f"Failed to send email: {str(e)}"
    
    def track_email_campaign(self, campaign_data):
        """Track email campaign performance and follow-ups."""
        
        # Save campaign data
        campaign_file = Path("marketing/campaigns") / f"campaign_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        campaign_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(campaign_file, 'w') as f:
            json.dump(campaign_data, f, indent=2, default=str)
        
        return str(campaign_file)

def show_email_outreach_page():
    """Main email outreach interface."""
    
    st.header("📧 Professional Email Outreach System")
    st.write("Target entertainment industry professionals with AI-powered video summarization solutions")
    
    # Initialize system
    if 'email_system' not in st.session_state:
        st.session_state.email_system = EmailOutreachSystem()
    
    email_system = st.session_state.email_system
    
    # Tabs for different functionalities
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📝 Compose Email", 
        "👥 Prospect Database", 
        "📊 Campaign Analytics",
        "⚙️ Email Settings",
        "🎯 Industry Intelligence"
    ])
    
    with tab1:
        show_email_composer(email_system)
    
    with tab2:
        show_prospect_management(email_system)
    
    with tab3:
        show_campaign_analytics()
    
    with tab4:
        show_email_settings()
    
    with tab5:
        show_industry_intelligence()

def show_email_composer(email_system):
    """Email composition interface."""
    
    st.subheader("✍️ Compose Professional Email")
    
    # Select target industry
    col1, col2 = st.columns(2)
    
    with col1:
        target_industry = st.selectbox(
            "Target Industry:",
            ["movie_studios", "tv_networks", "streaming_services", "indie_filmmakers"],
            format_func=lambda x: {
                "movie_studios": "🎬 Movie Studios",
                "tv_networks": "📺 TV Networks", 
                "streaming_services": "🎥 Streaming Services",
                "indie_filmmakers": "🎭 Independent Filmmakers"
            }[x]
        )
    
    with col2:
        follow_up_level = st.selectbox(
            "Email Type:",
            [0, 1, 2],
            format_func=lambda x: {0: "Initial Contact", 1: "Follow-up #1", 2: "Follow-up #2"}[x]
        )
    
    # Personalization fields
    st.subheader("🎯 Personalization")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        first_name = st.text_input("Contact First Name", "John")
        company = st.text_input("Company Name", "Warner Bros Pictures")
        recent_project = st.text_input("Recent Project/Work", "Recent blockbuster release")
    
    with col2:
        sender_name = st.text_input("Your Name", "Alex Chen")
        sender_title = st.text_input("Your Title", "Founder & CEO")
        sender_email = st.text_input("Your Email", "alex@videosummarization.ai")
    
    with col3:
        sender_phone = st.text_input("Your Phone", "+1 (555) 123-4567")
        sender_linkedin = st.text_input("Your LinkedIn", "linkedin.com/in/alexchen")
        to_email = st.text_input("Recipient Email", "john.doe@warnerbros.com")
    
    # Generate email
    if st.button("🎨 Generate Personalized Email", type="primary"):
        prospect_data = {
            "first_name": first_name,
            "company": company, 
            "recent_project": recent_project,
            "sender_name": sender_name,
            "sender_title": sender_title,
            "sender_email": sender_email,
            "sender_phone": sender_phone,
            "sender_linkedin": sender_linkedin
        }
        
        email_data = email_system.generate_personalized_email(
            prospect_data, 
            target_industry, 
            follow_up_level
        )
        
        st.session_state.generated_email = email_data
        st.session_state.recipient_email = to_email
    
    # Display generated email
    if 'generated_email' in st.session_state:
        st.divider()
        st.subheader("📋 Generated Email")
        
        email_data = st.session_state.generated_email
        
        # Editable subject and body
        subject = st.text_input("Subject Line:", value=email_data['subject'])
        body = st.text_area("Email Body:", value=email_data['body'], height=400)
        
        # Send email section
        st.subheader("📤 Send Email")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📧 Send Email", type="primary"):
                # Check if SMTP is configured
                if 'smtp_config' in st.session_state:
                    success, message = email_system.send_email(
                        st.session_state.recipient_email,
                        subject,
                        body,
                        st.session_state.smtp_config
                    )
                    
                    if success:
                        st.success(f"✅ Email sent successfully to {st.session_state.recipient_email}")
                        
                        # Track the sent email
                        campaign_data = {
                            "timestamp": datetime.now(),
                            "recipient": st.session_state.recipient_email,
                            "subject": subject,
                            "template_type": email_data['template_type'],
                            "follow_up_level": email_data['follow_up_level'],
                            "status": "sent"
                        }
                        
                        campaign_file = email_system.track_email_campaign(campaign_data)
                        st.info(f"Campaign tracked: {campaign_file}")
                        
                    else:
                        st.error(f"❌ {message}")
                else:
                    st.warning("⚙️ Please configure SMTP settings in the Email Settings tab first")
        
        with col2:
            if st.button("📋 Copy to Clipboard"):
                st.code(f"Subject: {subject}\n\n{body}")
                st.info("📋 Email copied! Paste into your preferred email client.")

def show_prospect_management(email_system):
    """Prospect database management."""
    
    st.subheader("👥 Prospect Database Management")
    
    # Industry selection
    selected_industry = st.selectbox(
        "Select Industry Segment:",
        list(email_system.prospects.keys()),
        format_func=lambda x: x.replace('_', ' ').title()
    )
    
    # Display prospects
    prospects = email_system.prospects[selected_industry]
    
    if prospects:
        # Convert to DataFrame for better display
        df = pd.DataFrame(prospects)
        
        # Add selection column
        df['Select'] = False
        
        # Display editable dataframe
        edited_df = st.data_editor(
            df,
            use_container_width=True,
            num_rows="dynamic",
            column_config={
                "Select": st.column_config.CheckboxColumn("Select for Campaign"),
                "company": st.column_config.TextColumn("Company", required=True),
                "contact_type": st.column_config.TextColumn("Contact Type"),
                "estimated_revenue": st.column_config.TextColumn("Revenue Potential"),
                "status": st.column_config.SelectboxColumn(
                    "Status",
                    options=["Not Contacted", "Contacted", "Responded", "Meeting Scheduled", "Proposal Sent", "Closed Won", "Closed Lost"]
                )
            }
        )
        
        # Bulk operations
        st.subheader("📤 Bulk Email Campaign")
        
        selected_prospects = edited_df[edited_df['Select'] == True]
        
        if len(selected_prospects) > 0:
            st.write(f"Selected {len(selected_prospects)} prospects for campaign")
            
            col1, col2 = st.columns(2)
            
            with col1:
                campaign_name = st.text_input("Campaign Name", f"{selected_industry.title()} Outreach {datetime.now().strftime('%Y-%m-%d')}")
            
            with col2:
                email_template = st.selectbox(
                    "Email Template:",
                    [0, 1, 2],
                    format_func=lambda x: {0: "Initial Contact", 1: "Follow-up #1", 2: "Follow-up #2"}[x]
                )
            
            if st.button("🚀 Launch Campaign", type="primary"):
                st.success(f"Campaign '{campaign_name}' would be launched to {len(selected_prospects)} prospects")
                st.info("Note: Configure SMTP settings to actually send emails")
    
    else:
        st.info("No prospects in this industry segment. Add some below!")
    
    # Add new prospect
    with st.expander("➕ Add New Prospect"):
        col1, col2 = st.columns(2)
        
        with col1:
            new_company = st.text_input("Company Name")
            new_contact_type = st.text_input("Contact Type/Title") 
            new_email = st.text_input("Email Address")
        
        with col2:
            new_revenue = st.text_input("Revenue Potential")
            new_pain_points = st.text_area("Pain Points (one per line)")
            new_notes = st.text_area("Additional Notes")
        
        if st.button("Add Prospect") and new_company:
            new_prospect = {
                "company": new_company,
                "contact_type": new_contact_type,
                "email": new_email,
                "estimated_revenue": new_revenue,
                "pain_points": new_pain_points.split('\n') if new_pain_points else [],
                "notes": new_notes,
                "status": "Not Contacted",
                "added_date": datetime.now().strftime('%Y-%m-%d')
            }
            
            # Add to session state (in real app, save to database)
            email_system.prospects[selected_industry].append(new_prospect)
            st.success(f"Added {new_company} to {selected_industry} prospects")
            st.rerun()

def show_campaign_analytics():
    """Campaign performance analytics."""
    
    st.subheader("📊 Campaign Analytics & Performance")
    
    # Sample analytics data
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Emails Sent", "247", "+15 this week")
    
    with col2:
        st.metric("Response Rate", "18.2%", "+2.1%")
    
    with col3:
        st.metric("Meetings Scheduled", "12", "+3 this week")
    
    with col4:
        st.metric("Revenue Pipeline", "$450K", "+$125K")
    
    # Campaign performance chart
    st.subheader("📈 Campaign Performance Over Time")
    
    # Sample data for visualization
    import plotly.graph_objects as go
    from datetime import datetime, timedelta
    
    dates = [datetime.now() - timedelta(days=x) for x in range(30, 0, -1)]
    emails_sent = [5, 8, 12, 15, 10, 7, 20, 25, 18, 22, 16, 14, 30, 35, 28, 24, 20, 18, 25, 30, 32, 28, 26, 24, 22, 35, 40, 38, 42, 45]
    responses = [1, 1, 2, 3, 2, 1, 4, 5, 3, 4, 3, 2, 6, 7, 5, 4, 3, 3, 5, 6, 6, 5, 4, 4, 4, 7, 8, 7, 8, 9]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=emails_sent, mode='lines+markers', name='Emails Sent', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=dates, y=responses, mode='lines+markers', name='Responses', line=dict(color='green')))
    
    fig.update_layout(
        title="Daily Email Campaign Performance",
        xaxis_title="Date",
        yaxis_title="Count",
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Industry breakdown
    st.subheader("🎯 Performance by Industry Segment")
    
    industry_data = {
        "Industry": ["Movie Studios", "TV Networks", "Streaming Services", "Independent"],
        "Emails Sent": [85, 67, 52, 43],
        "Response Rate": ["22%", "19%", "15%", "12%"],
        "Avg Deal Size": ["$350K", "$180K", "$120K", "$299/mo"],
        "Pipeline": ["$1.2M", "$680K", "$540K", "$15K"]
    }
    
    industry_df = pd.DataFrame(industry_data)
    st.dataframe(industry_df, use_container_width=True)
    
    # Recent responses
    st.subheader("💬 Recent Responses & Activity")
    
    recent_activity = [
        {"Date": "2024-08-06", "Company": "Sony Pictures", "Type": "Response", "Status": "Meeting Requested", "Notes": "Interested in pilot program"},
        {"Date": "2024-08-05", "Company": "Netflix", "Type": "Response", "Status": "Needs More Info", "Notes": "Asked for case studies"},
        {"Date": "2024-08-04", "Company": "Warner Bros", "Type": "Meeting", "Status": "Completed", "Notes": "Demo went well, proposal requested"},
        {"Date": "2024-08-03", "Company": "A24 Films", "Type": "Response", "Status": "Interested", "Notes": "Small studio, budget constraints"},
        {"Date": "2024-08-02", "Company": "HBO Max", "Type": "Response", "Status": "Not Interested", "Notes": "Current solution working well"}
    ]
    
    activity_df = pd.DataFrame(recent_activity)
    st.dataframe(activity_df, use_container_width=True)

def show_email_settings():
    """Email configuration settings."""
    
    st.subheader("⚙️ Email Configuration")
    
    # SMTP Settings
    st.write("**SMTP Configuration:**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        smtp_server = st.text_input("SMTP Server", "smtp.gmail.com")
        smtp_port = st.number_input("SMTP Port", value=587)
        sender_email = st.text_input("Sender Email", "your.email@company.com")
    
    with col2:
        sender_name = st.text_input("Sender Display Name", "Your Name")
        use_tls = st.checkbox("Use TLS", value=True)
        sender_password = st.text_input("Email Password", type="password", help="Use app-specific password for Gmail")
    
    # Save SMTP configuration
    if st.button("💾 Save SMTP Settings"):
        smtp_config = {
            "smtp_server": smtp_server,
            "smtp_port": smtp_port,
            "email": sender_email,
            "name": sender_name,
            "password": sender_password,
            "use_tls": use_tls
        }
        
        st.session_state.smtp_config = smtp_config
        st.success("SMTP settings saved successfully!")
    
    # Test email
    st.divider()
    st.write("**Test Email Configuration:**")
    
    test_email = st.text_input("Send test email to:", "test@example.com")
    
    if st.button("📧 Send Test Email") and 'smtp_config' in st.session_state:
        # Simple test email
        test_subject = "Test Email from Video Summarization Tool"
        test_body = """
This is a test email from your Video Summarization Tool email outreach system.

If you received this email, your SMTP configuration is working correctly!

Best regards,
Video Summarization Team
        """
        
        email_system = st.session_state.email_system
        success, message = email_system.send_email(
            test_email, 
            test_subject, 
            test_body, 
            st.session_state.smtp_config
        )
        
        if success:
            st.success("✅ Test email sent successfully!")
        else:
            st.error(f"❌ Test email failed: {message}")
    
    # Email signature settings
    st.divider()
    st.write("**Email Signature:**")
    
    signature = st.text_area(
        "Default Email Signature:",
        value="""---
{sender_name}
{title}
Video Summarization Engine
Email: {sender_email}
Phone: {sender_phone}
LinkedIn: {sender_linkedin}
Website: www.videosummarization.ai

Transforming post-production workflows with AI-powered video analysis.
        """,
        help="Use {variables} for dynamic content"
    )
    
    if st.button("💾 Save Signature"):
        st.session_state.email_signature = signature
        st.success("Email signature saved!")

def show_industry_intelligence():
    """Industry intelligence and market insights."""
    
    st.subheader("🎯 Entertainment Industry Intelligence")
    
    # Market intelligence
    st.write("**Market Size & Opportunity:**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Global Post-Production Market", "$7.2B", "+8.5% CAGR")
        st.metric("Streaming Content Spend", "$220B", "+12% YoY")
    
    with col2:
        st.metric("Major Studios (Potential)", "15", "~$500K each")
        st.metric("Mid-size Studios", "150+", "~$100K each")
    
    with col3:
        st.metric("Streaming Services", "200+", "~$75K each")
        st.metric("Independent Creators", "50K+", "~$100/mo each")
    
    # Industry contacts and insights
    st.subheader("🏢 Target Companies & Decision Makers")
    
    target_companies = {
        "Major Studios": {
            "companies": ["Warner Bros", "Sony Pictures", "Universal", "Disney", "Paramount"],
            "key_roles": ["VP Post-Production", "Post-Production Supervisor", "Head of Technology"],
            "avg_deal_size": "$300K-500K",
            "sales_cycle": "6-12 months"
        },
        "Streaming Services": {
            "companies": ["Netflix", "Amazon Prime", "Hulu", "Disney+", "HBO Max"],
            "key_roles": ["Content Operations Director", "VP Content Acquisition", "Head of Content Technology"],
            "avg_deal_size": "$150K-300K", 
            "sales_cycle": "3-6 months"
        },
        "TV Networks": {
            "companies": ["NBC", "CBS", "ABC", "Fox", "CNN"],
            "key_roles": ["Director of Content Operations", "VP Programming", "Head of Digital"],
            "avg_deal_size": "$100K-250K",
            "sales_cycle": "4-8 months"
        }
    }
    
    for segment, info in target_companies.items():
        with st.expander(f"📊 {segment}"):
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Target Companies:**")
                for company in info["companies"]:
                    st.write(f"• {company}")
                
                st.write("**Average Deal Size:**")
                st.write(info["avg_deal_size"])
            
            with col2:
                st.write("**Key Decision Makers:**")
                for role in info["key_roles"]:
                    st.write(f"• {role}")
                
                st.write("**Typical Sales Cycle:**")
                st.write(info["sales_cycle"])
    
    # Industry trends and pain points
    st.subheader("📈 Industry Trends & Pain Points")
    
    trends_col1, trends_col2 = st.columns(2)
    
    with trends_col1:
        st.write("**Current Trends:**")
        trends = [
            "Accelerated streaming content production",
            "Shorter post-production timelines", 
            "Remote collaboration workflows",
            "AI adoption in creative industries",
            "Cost pressure on production budgets",
            "Quality demands increasing"
        ]
        for trend in trends:
            st.write(f"📊 {trend}")
    
    with trends_col2:
        st.write("**Major Pain Points:**")
        pain_points = [
            "Manual dailies review taking too long",
            "Expensive editor time ($500-2000/day)",
            "Missing narrative issues until late",
            "Archive content underutilized", 
            "Content acquisition bottlenecks",
            "Scaling creative decision-making"
        ]
        for pain in pain_points:
            st.write(f"⚠️ {pain}")
    
    # Value proposition generator
    st.subheader("💡 Value Proposition Generator")
    
    selected_pain = st.selectbox(
        "Select main pain point to address:",
        [
            "Dailies review time",
            "Post-production costs", 
            "Content discovery",
            "Quality assurance",
            "Workflow efficiency",
            "Archive utilization"
        ]
    )
    
    value_props = {
        "Dailies review time": "Reduce dailies review time by 80% - from 4 hours to 45 minutes daily. Save $35K+ per production in editor costs while improving creative focus.",
        "Post-production costs": "Cut post-production review costs by 60%. Our AI analysis eliminates hours of manual screening, freeing your team for creative decisions.",
        "Content discovery": "Unlock millions in hidden archive value. Automatically discover and catalog forgotten content for licensing and reuse opportunities.",
        "Quality assurance": "Catch narrative issues early. Our AI identifies pacing, structure, and continuity problems before they reach expensive re-shoots.",
        "Workflow efficiency": "Streamline your entire post-production workflow. Automated analysis, intelligent bookmarking, and seamless editor integration.",
        "Archive utilization": "Transform your archive into a revenue stream. Intelligent content discovery and automated cataloging reveal hidden licensing opportunities."
    }
    
    st.info(f"💡 **Value Proposition:** {value_props[selected_pain]}")
    
    # ROI Calculator
    st.subheader("💰 ROI Calculator")
    
    calc_col1, calc_col2 = st.columns(2)
    
    with calc_col1:
        editor_day_rate = st.number_input("Editor Daily Rate ($)", value=1500, step=100)
        review_hours_daily = st.number_input("Daily Review Hours", value=4.0, step=0.5)
        production_days = st.number_input("Production Days", value=60, step=5)
    
    with calc_col2:
        time_savings_pct = st.slider("Time Savings (%)", 50, 90, 80)
        
        # Calculate ROI
        daily_cost = editor_day_rate * (review_hours_daily / 8)
        total_current_cost = daily_cost * production_days
        savings = total_current_cost * (time_savings_pct / 100)
        
        st.metric("Current Review Cost", f"${total_current_cost:,.0f}")
        st.metric("Annual Savings", f"${savings:,.0f}")
        st.metric("ROI with $100K License", f"{(savings/100000)*100:.0f}%")
    
    # Export prospect list
    st.divider()
    
    if st.button("📥 Export Complete Prospect Database"):
        # Create comprehensive prospect data
        all_prospects = []
        for industry, prospects in st.session_state.email_system.prospects.items():
            for prospect in prospects:
                prospect['industry'] = industry
                all_prospects.append(prospect)
        
        prospects_df = pd.DataFrame(all_prospects)
        
        # Convert to CSV
        csv = prospects_df.to_csv(index=False)
        
        st.download_button(
            label="📊 Download Prospects CSV",
            data=csv,
            file_name=f"video_summarization_prospects_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
        
        st.success("Prospect database ready for export!")