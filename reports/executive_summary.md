# Executive Summary: Customer Churn Analysis

## Business Problem
In the highly competitive telecommunications sector, customer churn represents one of the most significant leaks in annualized recurring revenue (ARR). Acquiring a new customer is significantly more expensive than retaining an existing one. Our objective was to prospectively identify at-risk customers, understand the core drivers of attrition, quantify the revenue at risk, and propose targeted retention strategies to improve the bottom line.

## Key Findings
Through exploratory data analysis and predictive modeling, we analyzed a cohort of over 7,000 customers. Key findings underscore that churn is not random, but highly conditional on contract structure and early consumer experiences:
- **Baseline Churn Magnitude:** Approximately 26.5% of the customer base is currently churning.
- **Contract Type is the Dominant Factor:** Customers on flexible, month-to-month contracts exhibit drastically higher churn rates compared to those on 1-year or 2-year subscriptions.
- **Early-stage Attrition:** Churn risk peaks within the first 12 months (0-1 Year cohort) of a customer's lifecycle and sharply declines as tenure increases.
- **Service Dependency:** Customers lacking 'sticky' services (like Fiber Optic internet or integrated tech support) are more susceptible to leaving.

## Churn Drivers
A Logistic Regression model (chosen for its robust interpretability) was developed to extract the exact weight of individual business drivers. The top quantitative drivers increasing churn risk are:
1. **Month-to-month contracts**
2. **Fiber optic internet service** (often correlated with higher costs or service reliability issues)
3. **Electronic Check payment methods**

Conversely, features that strongly decrease churn risk include:
1. **Two-year contracts**
2. **Longer tenure**
3. **Subscriptions to bundled services** (e.g., DSL, Tech Support, Online Security)

## Revenue at Risk
By applying our predictive model to the current customer base, we can quantify the immediate financial exposure:
- **Total Monthly Revenue Analyzed:** ~$456,000 / month
- **Monthly Revenue Flagged as High-Risk:** ~$139,000 / month
- **Percentage of Revenue at Risk:** ~30.5% of total recurring revenue

*Note: These figures are based on the analyzed dataset and highlight the critical need for proactive intervention.*

## Retention Opportunities
We simulated the ROI of a targeted retention campaign focusing on our highest-value operational targets: customers flagged as high flight-risk who are not locked into long-term contracts. 

**Proposed Strategy:** Offer a proactive 10% discount to high-risk month-to-month customers if they agree to upgrade to a 1-year contract.

## Expected Business Impact
Based on a conservative 30% campaign acceptance rate, our simulations indicate:
- **Projected Drop in Churn Probability:** The commitment of a 1-year contract significantly drops the algorithmic risk of churn.
- **Estimated Savings:** Up to **$33,000 / month** in retained revenue (roughly $400,000 annualized).
- **Secondary Benefits:** Increased customer lifetime value (LTV) as customers pushed past the 1-year mark naturally exhibit lower subsequent churn rates.

## Conclusion
Customer attrition is structural and heavily dependent on contract types and early-tenure engagement. By operationalizing these insights and shifting from reactive to predictive retention strategies, the business can protect a substantial portion of its recurring revenue.
