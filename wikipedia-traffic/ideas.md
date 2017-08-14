- Explore pyflux
- Trend (think about outliers and period)
- Are we able to see mobile/web/aggregate traffic separately for all?
- Spike elimination
- is fillna(0) optimal?
- Feature ideas:
  + Language/Agent type/Access type/resource
  + Aggregate features
  + Rolling median/std for a range of periods
  + Some metric indicating number of 0 visits in the past
  + Clustering?
  + Regional holiday
  + Weekday/weekend medians
- Holt Winters + GAS models + seasonal for weekly oscillation removal?
- Detect seasonality with https://www.r-bloggers.com/detecting-seasonality/ rather than a FFT?
- Try neural nets
