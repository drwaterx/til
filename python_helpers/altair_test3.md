<html>
<head>
  <link href="styles.css" rel="stylesheet" />
  <script src="https://cdn.jsdelivr.net/npm/vega@{{ alt.VEGA_VERSION }}"></script>
  <script src="https://cdn.jsdelivr.net/npm/vega-lite@{{ alt.VEGALITE_VERSION }}"></script>
  <script src="https://cdn.jsdelivr.net/npm/vega-embed@{{ alt.VEGAEMBED_VERSION }}"></script>
</head>

<body>
<h1>Model performance dashboard</h1>
<h2>{{ subtitle }}</h2>

<p>
{{ dchart1_title }}
</p>

	<div id="vis"></div>
      <script type="text/javascript">
      var spec = {{ dchart1 }};  /* var chart = {{ chart|safe }}; */
      vegaEmbed('#vis', spec);
      </script>
</body>
</html>