"""
Deployment options and pricing calculator for MovieDigest AI
Helps determine optimal deployment strategy based on customer needs
"""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime

def show_deployment_options_page():
    """Display deployment strategy and pricing options."""
    
    st.header("🚀 Deployment Strategy for MovieDigest AI")
    st.write("Serving offline video analysis to online customers")
    
    # Deployment model selector
    st.subheader("🎯 Choose Your Deployment Model")
    
    deployment_models = {
        "Hybrid SaaS + Desktop": {
            "description": "Web dashboard for management + downloadable desktop app for processing",
            "target_audience": "Individual creators, small studios",
            "pricing": "$99-299/month per user",
            "advantages": ["Data stays local", "Easy updates", "Web-based billing"],
            "implementation_time": "2-3 months"
        },
        "Enterprise On-Premises": {
            "description": "Custom deployment at customer facilities with cloud management",
            "target_audience": "Major studios, streaming platforms",
            "pricing": "$100K-500K setup + $200K-1M annual",
            "advantages": ["Complete control", "Custom integration", "Dedicated support"],
            "implementation_time": "4-6 months"
        },
        "Containerized Self-Hosted": {
            "description": "Docker containers for easy deployment on customer infrastructure",
            "target_audience": "Tech-savvy studios, production companies",
            "pricing": "$999-5K/month license",
            "advantages": ["Easy deployment", "Scalable", "Infrastructure control"],
            "implementation_time": "1-2 months"
        },
        "Desktop Application": {
            "description": "Packaged desktop software with web-based licensing",
            "target_audience": "Independent creators, freelancers",
            "pricing": "$299-999/month subscription",
            "advantages": ["Simple installation", "Offline capable", "Familiar interface"],
            "implementation_time": "3-4 months"
        }
    }
    
    selected_model = st.selectbox(
        "Select deployment model to analyze:",
        list(deployment_models.keys())
    )
    
    model_info = deployment_models[selected_model]
    
    # Display model details
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Description:**")
        st.info(model_info["description"])
        
        st.write("**Target Audience:**")
        st.write(model_info["target_audience"])
        
        st.write("**Pricing Range:**")
        st.success(model_info["pricing"])
    
    with col2:
        st.write("**Key Advantages:**")
        for advantage in model_info["advantages"]:
            st.write(f"✅ {advantage}")
        
        st.write("**Implementation Time:**")
        st.warning(model_info["implementation_time"])
    
    # Revenue projection calculator
    st.subheader("💰 Revenue Projection Calculator")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**Customer Segments:**")
        individual_users = st.number_input("Individual Users", min_value=0, value=50)
        individual_price = st.number_input("Price per Individual ($)", min_value=0, value=99)
        
        small_studios = st.number_input("Small Studios", min_value=0, value=10)
        studio_price = st.number_input("Price per Studio ($)", min_value=0, value=2500)
    
    with col2:
        st.write("**Enterprise Customers:**")
        enterprise_clients = st.number_input("Enterprise Clients", min_value=0, value=3)
        enterprise_price = st.number_input("Enterprise Annual ($)", min_value=0, value=200000)
        
        setup_fees = st.number_input("Setup Fees Total ($)", min_value=0, value=150000)
    
    with col3:
        st.write("**Growth Assumptions:**")
        months_projection = st.slider("Projection Period (months)", 6, 36, 12)
        monthly_growth_rate = st.slider("Monthly Growth Rate (%)", 0, 20, 10) / 100
        churn_rate = st.slider("Monthly Churn Rate (%)", 0, 10, 5) / 100
    
    # Calculate projections
    monthly_recurring = (
        individual_users * individual_price +
        small_studios * studio_price +
        enterprise_clients * (enterprise_price / 12)
    )
    
    annual_recurring = monthly_recurring * 12
    total_with_setup = annual_recurring + setup_fees
    
    # Display results
    st.subheader("📊 Revenue Projections")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Monthly Recurring Revenue", f"${monthly_recurring:,.0f}")
    
    with col2:
        st.metric("Annual Recurring Revenue", f"${annual_recurring:,.0f}")
    
    with col3:
        st.metric("Year 1 Total (with setup)", f"${total_with_setup:,.0f}")
    
    with col4:
        projected_year_2 = annual_recurring * (1 + monthly_growth_rate * 12)
        st.metric("Projected Year 2 ARR", f"${projected_year_2:,.0f}")
    
    # Growth projection chart
    months = list(range(1, months_projection + 1))
    recurring_revenue = []
    cumulative_revenue = []
    
    current_customers = individual_users + small_studios + enterprise_clients
    current_mrr = monthly_recurring
    total_revenue = setup_fees
    
    for month in months:
        # Apply growth and churn
        current_customers *= (1 + monthly_growth_rate - churn_rate)
        current_mrr = monthly_recurring * (current_customers / (individual_users + small_studios + enterprise_clients))
        
        recurring_revenue.append(current_mrr)
        total_revenue += current_mrr
        cumulative_revenue.append(total_revenue)
    
    # Create projection chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=months,
        y=recurring_revenue,
        mode='lines+markers',
        name='Monthly Recurring Revenue',
        line=dict(color='blue')
    ))
    
    fig.add_trace(go.Scatter(
        x=months,
        y=cumulative_revenue,
        mode='lines+markers',
        name='Cumulative Revenue',
        line=dict(color='green'),
        yaxis='y2'
    ))
    
    fig.update_layout(
        title="Revenue Growth Projection",
        xaxis_title="Months",
        yaxis_title="Monthly Recurring Revenue ($)",
        yaxis2=dict(
            title="Cumulative Revenue ($)",
            overlaying='y',
            side='right'
        ),
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Deployment comparison
    st.subheader("📋 Deployment Model Comparison")
    
    comparison_data = {
        "Model": ["Hybrid SaaS", "Enterprise On-Premises", "Containerized", "Desktop App"],
        "Setup Time": ["2-3 months", "4-6 months", "1-2 months", "3-4 months"],
        "Target Revenue": ["$500K ARR", "$2M ARR", "$1M ARR", "$300K ARR"],
        "Customer Effort": ["Low", "High", "Medium", "Low"],
        "Data Security": ["High", "Highest", "High", "Highest"],
        "Scalability": ["High", "Medium", "High", "Low"],
        "Support Complexity": ["Medium", "High", "Medium", "Low"]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True)
    
    # Implementation roadmap
    st.subheader("🗓️ Implementation Roadmap")
    
    roadmap_phases = {
        "Phase 1: Web Dashboard (Month 1-2)": [
            "Deploy marketing site on Replit",
            "Build user authentication and billing",
            "Create email marketing system",
            "Implement license key generation"
        ],
        "Phase 2: Desktop Client (Month 3-4)": [
            "Package Streamlit app with PyInstaller",
            "Implement licensing and activation",
            "Build automatic update mechanism",
            "Create installation packages"
        ],
        "Phase 3: Enterprise Solutions (Month 5-6)": [
            "Create Docker containers",
            "Build enterprise authentication",
            "Develop monitoring tools",
            "Create deployment documentation"
        ],
        "Phase 4: Scale & Optimize (Month 7+)": [
            "Expand email marketing campaigns",
            "Build customer success programs",
            "Add advanced analytics features",
            "Develop partner integration APIs"
        ]
    }
    
    for phase, tasks in roadmap_phases.items():
        with st.expander(phase):
            for task in tasks:
                st.write(f"• {task}")
    
    # Competitive analysis
    st.subheader("⚔️ Competitive Positioning")
    
    competitors = {
        "Traditional Video Editing Software": {
            "advantages_over": ["AI-powered analysis", "Automated summarization", "Web-based management"],
            "pricing_comparison": "Similar pricing but much more automated"
        },
        "Cloud-Based Video Analysis": {
            "advantages_over": ["Complete data privacy", "No upload requirements", "Local GPU acceleration"],
            "pricing_comparison": "Higher upfront cost but lower ongoing costs"
        },
        "Manual Review Processes": {
            "advantages_over": ["75% time savings", "Consistent analysis", "Scalable processing"],
            "pricing_comparison": "Replaces expensive human review time"
        }
    }
    
    selected_competitor = st.selectbox(
        "Compare against competitor type:",
        list(competitors.keys())
    )
    
    competitor_info = competitors[selected_competitor]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Our Advantages:**")
        for advantage in competitor_info["advantages_over"]:
            st.write(f"✅ {advantage}")
    
    with col2:
        st.write("**Pricing Position:**")
        st.info(competitor_info["pricing_comparison"])
    
    # Action plan generator
    st.subheader("📋 Next Steps Action Plan")
    
    priority_level = st.selectbox(
        "What's your priority?",
        ["Quick Market Entry", "Maximum Revenue", "Lowest Risk", "Enterprise Focus"]
    )
    
    action_plans = {
        "Quick Market Entry": {
            "timeline": "3 months",
            "focus": "Desktop Application model",
            "first_steps": [
                "Package current Streamlit app with PyInstaller",
                "Create simple licensing system",
                "Launch beta program with existing contacts",
                "Build payment processing integration"
            ]
        },
        "Maximum Revenue": {
            "timeline": "6 months", 
            "focus": "Enterprise On-Premises model",
            "first_steps": [
                "Create enterprise sales materials",
                "Build custom demo environment",
                "Target top 10 studios with personal outreach",
                "Develop proof-of-concept implementations"
            ]
        },
        "Lowest Risk": {
            "timeline": "4 months",
            "focus": "Hybrid SaaS + Desktop model",
            "first_steps": [
                "Start with web dashboard on Replit",
                "Create freemium tier for validation",
                "Build email list through content marketing",
                "Develop desktop app iteratively"
            ]
        },
        "Enterprise Focus": {
            "timeline": "8 months",
            "focus": "Containerized + On-Premises models",
            "first_steps": [
                "Create enterprise-grade containerization",
                "Build comprehensive security documentation",
                "Develop custom integration capabilities",
                "Establish enterprise sales process"
            ]
        }
    }
    
    plan = action_plans[priority_level]
    
    st.write(f"**Recommended Approach: {plan['focus']}**")
    st.write(f"**Timeline: {plan['timeline']}**")
    
    st.write("**First Steps:**")
    for i, step in enumerate(plan['first_steps'], 1):
        st.write(f"{i}. {step}")
    
    # Download action plan
    if st.button("📥 Download Complete Strategy Document"):
        strategy_content = f"""
# MovieDigest AI Deployment Strategy

## Selected Model: {selected_model}
{model_info['description']}

## Revenue Projections
- Monthly Recurring Revenue: ${monthly_recurring:,.0f}
- Annual Recurring Revenue: ${annual_recurring:,.0f}
- Year 1 Total Revenue: ${total_with_setup:,.0f}

## Priority: {priority_level}
Timeline: {plan['timeline']}
Focus: {plan['focus']}

## Next Steps:
{chr(10).join([f"{i}. {step}" for i, step in enumerate(plan['first_steps'], 1)])}

## Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        st.download_button(
            label="📋 Download Strategy",
            data=strategy_content,
            file_name=f"MovieDigest_Deployment_Strategy_{datetime.now().strftime('%Y%m%d')}.md",
            mime="text/markdown"
        )

if __name__ == "__main__":
    show_deployment_options_page()