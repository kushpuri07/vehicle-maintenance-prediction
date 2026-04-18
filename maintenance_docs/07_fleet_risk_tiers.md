# FLEET RISK CLASSIFICATION AND PRIORITIZATION

## Overview
When managing a fleet of vehicles, individual maintenance needs must be prioritized against each other. This document defines standardized risk tiers and timelines so that limited service capacity can be allocated effectively.

## Risk Tiers

### CRITICAL (Tier 1)
**Definition**: Vehicle poses immediate safety risk or is likely to fail soon.

**Triggers (any one qualifies)**:
- Brake condition is "Worn Out"
- Reported issues is 5
- Model-predicted maintenance need with probability ≥ 95% AND at least one component in worst state
- Multiple worst-state components (2 or more of: Worn Out tires, Worn Out brakes, Weak battery)

**Action**: Remove from active duty until serviced. Service within 48 hours.

### HIGH (Tier 2)
**Definition**: Elevated risk; service within 2 weeks to prevent escalation.

**Triggers (any one qualifies)**:
- Any single component in worst state (Worn Out / Weak) that isn't already Critical
- Reported issues of 3 or 4
- Model-predicted maintenance need with probability 80–94%
- Vehicle age 10+ years AND maintenance history is "Poor"
- Accident history of 3+

**Action**: Schedule service within 2 weeks. Vehicle may remain in light duty.

### MEDIUM (Tier 3)
**Definition**: Noticeable concerns; service at next scheduled opportunity or within 1 month.

**Triggers (any one qualifies)**:
- Reported issues of 2
- Maintenance history is "Average" or "Poor"
- Vehicle age 6–9 years with no Critical/High triggers
- Model-predicted maintenance need with probability 60–79%

**Action**: Include in next planned service cycle.

### LOW (Tier 4)
**Definition**: Healthy vehicle; routine monitoring only.

**Triggers**:
- All components rated "Good" or "New"
- Reported issues of 0 or 1
- Model-predicted maintenance need with probability below 60%
- No accident or maintenance history concerns

**Action**: Follow manufacturer's scheduled maintenance plan. No intervention.

## Prioritization Framework

When scheduling servicing for a fleet, use this order:

1. All CRITICAL vehicles first — no exceptions. These vehicles should be pulled from duty immediately.
2. HIGH-tier vehicles next, in order of:
   a. Safety-critical components (brakes > tires > battery > other)
   b. Reported issue count (higher = sooner)
   c. Vehicle age (older = sooner)
3. MEDIUM-tier vehicles rolled into normal service windows.
4. LOW-tier vehicles serviced on manufacturer schedule.

## Cost-Benefit Guidance

For each vehicle, cost of deferred maintenance compounds approximately 15–25% per month the issue is ignored. Critical issues compound faster.

| Tier | Typical Service Cost | Cost if Deferred 3 Months |
|---|---|---|
| Critical | ₹8,000 – ₹25,000 | ₹20,000 – ₹80,000 (plus downtime) |
| High | ₹5,000 – ₹15,000 | ₹10,000 – ₹30,000 |
| Medium | ₹3,000 – ₹10,000 | ₹5,000 – ₹15,000 |
| Low | ₹1,500 – ₹5,000 | Same if still low tier |

## Fleet-Level Metrics to Track

- **Critical ratio**: % of fleet in CRITICAL tier at any time. Target: below 5%.
- **High+Critical ratio**: Target below 15%.
- **Average reported issues per vehicle**: Rising trend indicates systemic fleet aging.
- **Service compliance rate**: % of recommended services actually performed. Target 95%+.

## Escalation Rules

- Any vehicle with 2+ tier escalations in 6 months should receive a comprehensive inspection regardless of current tier.
- Any vehicle categorized CRITICAL twice in 12 months should be evaluated for end-of-service-life.
- Fleet-wide trend of rising Critical tier count indicates a need to audit service vendors or operating conditions.

## Safety Notes

- Never override a CRITICAL classification based on operational pressure. A vehicle failure on duty is always more expensive than a day out of service.
- If multiple vehicles are Critical simultaneously and capacity is limited, prioritize by passenger count (buses > vans > cars > motorcycles) and then by brake-related issues first.
- Record every tier change in the fleet management log for trend analysis.
