# SAFETY DISCLAIMERS AND OPERATIONAL NOTICES

## Purpose of This Document
This document defines the safety disclaimers and operational boundaries that must accompany any automated maintenance recommendation. These statements should be surfaced to the end user in every report.

## Standard Disclaimer (Required on All Reports)

> **Disclaimer**: This maintenance assessment is generated based on the data provided and is intended as a decision-support tool for fleet operators. It does not replace on-site inspection by a certified automotive technician. Actual vehicle condition may differ from input data. Safety-critical services (brakes, steering, tires, airbags) must be verified and performed by qualified professionals. The operator assumes full responsibility for servicing decisions and their consequences.

## Scope Boundaries

This system **can** advise on:
- Priority and timing of maintenance services
- Likely components needing attention based on input data
- Cost estimates for reference
- Comparison of risk across multiple vehicles in a fleet

This system **cannot**:
- Diagnose specific mechanical faults without physical inspection
- Replace OBD-II or dealer-specific diagnostic tools
- Certify a vehicle as roadworthy
- Predict sudden catastrophic failures with certainty
- Verify that service work was performed correctly

## Data Quality Caveats

Recommendations are only as accurate as the input data. If any of these conditions are true, recommendations should be treated as indicative only:
- Component conditions were not inspected recently
- Reported issues were self-reported rather than diagnosed
- Service history records are incomplete
- Odometer reading is believed to be inaccurate

## Regional Compliance

Maintenance standards vary by region:
- Legal minimum tire tread depth varies (1.6mm common; some regions require more)
- Emissions inspection intervals vary by country/state
- Commercial vehicle safety inspections have jurisdiction-specific requirements
- Insurance claims related to deferred maintenance may be denied if records are incomplete

Operators are responsible for ensuring recommendations comply with local regulations.

## Safety-Critical Override

Any vehicle with a "Worn Out" brake condition, or 5 reported issues, or multiple concurrent worst-state components should be treated as requiring immediate inspection regardless of any other analysis. These conditions represent patterns commonly associated with accident risk.

## Electric Vehicle Notice

Maintenance recommendations in this system primarily address internal combustion (petrol/diesel) and standard electric vehicles at a general level. EV-specific services (traction battery cooling, regenerative brake calibration, high-voltage system inspection) require manufacturer-certified service centers and are outside the scope of general fleet analysis.

## Commercial Fleet Notice

For commercial fleets (trucks, buses, delivery vans), additional regulatory inspection regimes may apply. Regular fleet analytics should not substitute for:
- Statutory annual inspections
- Driver pre-trip inspection checklists
- Tachograph and logbook compliance
- Tire re-treading standards for commercial use

## Liability Statement

Automated maintenance recommendations are advisory. Final decisions on vehicle servicing, return to duty, or retirement rest with the fleet operator and their certified technicians. Neither the recommendation system nor its operators assume liability for the operational consequences of servicing decisions made based on these outputs.

## Emergency Situations

If a vehicle exhibits any of the following during operation, remove it from service immediately regardless of scheduled recommendations:
- Loss of braking power or unusual brake pedal behavior
- Sudden loss of power steering
- Tire blowout or sidewall separation
- Smoke, fire, or fuel leak
- Electrical fire or persistent electrical smell
- Engine warning lights accompanied by performance loss

These situations require immediate professional intervention and are outside the scope of predictive maintenance.
