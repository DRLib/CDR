colorList = d3.scaleOrdinal(d3.schemeCategory10);
scatterModel1 = new NewScatterModel('#project_view_scatter');
linkModel = new LinkModel();
paraModel = new ParallelModel("#parallel_view_plot");
contourModel = new ContourModel();
//color_reference: https://www.d3js.org.cn/document/d3-scale-chromatic/#schemeCategory10
var color_reference = d3.schemeCategory10;
color_reference = color_reference.concat(color_reference[0]);

var node_colors = {
    "not_lasso": "lightgray", "hover": color_reference[3],
    "normal": "#ababab", "lasso": "#5a8ebb", "lasso_line": "#cecece"
}

var link_colors = {"cannot": color_reference[1], "must": color_reference[3]}
label = 0;
$.ajax({
    url: 'load_dataset_list',
    type: "GET",
    dataType: "json",
    data: "",
    success: function (result) {
        $.each(result.data, function (idx, item) {
            para_object.dataset_list.push(item)
        });
    }
});
sample_id = []
for (let i = 0; i < 10; i++) {
    sample_id.push(Math.floor(Math.random() * (100 + 1)))
}

