---
name: chart
description: >-
  Create data visualizations. Gets data via execute_sql, then renders
  a chart via create_chart_app. Both tools via MCP.
metadata:
  author: imrankha
  version: "2.0.0"
---
# chart

Do not call `task`.

## 1. Get the data

If you already have query results from a data-query step in this conversation,
use those results. Otherwise, read `/skills/data-query/SKILL.md` and follow it
first to get the data via `execute_sql`.

## 2. Create the chart

Call `create_chart_app` once with the data.

- Pick the best chart type: bar, line, scatter, pie, histogram, area, table,
  bubble, waterfall, funnel, treemap, radar, sankey, sunburst, gauge, heatmap,
  box_plot, violin.
- Always include a descriptive `title`.
- Set `xlabel` and `ylabel` when the chart has axes.
- Time series: use `line` or `area`.
- Comparisons: use `bar`.
- Proportions: use `pie` or `treemap`.
- Large datasets (20+ rows): aggregate first or use `histogram`.

## 3. Answer

Present the chart and one sentence describing what it shows.
If the raw data is also useful, mention it is available.
