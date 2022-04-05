function ContourModel() {

    this.draw_contour = function (svg_id, select_node_list) {
        let svg = d3.select(svg_id);
        let width = 130;
        let height = 130;
        let margin = {top: 10, right: 10, bottom: 10, left: 10};
        let embeddings = scatterModel1.data
        let link_type = svg_id.substring(1, 2)
        svg.attr("width", width).attr("height", height);
        svg.selectAll('g').remove();
        // prepare g
        let contour_canvas = svg.append('g');
        let contour_node_canvas = svg.append('g');
        let contour_link_canvas = svg.append('g');
        // prepare scale
        const max = {};
        const min = {};
        max.x = d3.max(embeddings, d => d[0]);
        max.y = d3.max(embeddings, d => d[1]);
        min.x = d3.min(embeddings, d => d[0]);
        min.y = d3.min(embeddings, d => d[1]);
        xScale = d3
            .scaleLinear()
            .domain([min.x, max.x])
            .range([margin.left, width - margin.right]);
        yScale = d3
            .scaleLinear()
            .domain([-max.y, -min.y])
            .range([margin.top, height - margin.bottom]);
        // prepare data
        let contourFunc = d3
            .contourDensity()
            .x(function (d) {
                return xScale(d[0]);
            })
            .y(function (d) {
                return yScale(-d[1]);
            })
            .bandwidth(5)
            .thresholds(10)

        let contourMapData = contourFunc(embeddings);

        function ticks(start, end, count) {
            let result = [],
                increment = (end - start) / count;
            for (let i = 0; i <= count; i++) {
                result.push(start + i * increment);
            }
            return result;
        }

        let colorBlue = d3
            .scalePow()
            .domain(ticks(0, d3.max(contourMapData.map((d) => d.value)), 3))
            .range(["#ffffff", "#b3cddd"]);

        var data = contourMapData.map((d) => {
            return {
                path: d3.geoPath()(d),
                color: colorBlue(d.value),
            };
        });
        //draw
        contour_canvas
            .selectAll("path")
            .data(data)
            .enter()
            .append("path")

        contour_canvas.exit().remove();

        contour_canvas
            .selectAll("path")
            .attr("stroke", "steelblue")
            .attr("stroke-width",0.25)
            .attr("d", (d) => d.path)
            .attr("fill", (d) => d.color);

        let x1 = embeddings[select_node_list[0]][0];
        let y1 = embeddings[select_node_list[0]][1];
        let x2 = embeddings[select_node_list[1]][0];
        let y2 = embeddings[select_node_list[1]][1];

        contour_node_canvas.append("circle")
            .attr("cx", d => xScale(x1))
            .attr("cy", d => yScale(-y1))
            .attr("r", 2)
            .attr("fill", 'black')
            .attr("stroke-width", "1px");

        contour_node_canvas.append("circle")
            .attr("cx", d => xScale(x2))
            .attr("cy", d => yScale(-y2))
            .attr("r", 2)
            .attr("fill", 'black')
            .attr("stroke-width", "1px");

        contour_link_canvas.append("line")
            .attr("x1", xScale(x1))
            .attr("y1", yScale(-y1))
            .attr("x2", xScale(x2))
            .attr("y2", yScale(-y2))
            // .attr('link_type',link_type)
            .attr("stroke", function (d) {
                if (link_type == "M") return link_colors['must'];
                else if (link_type === 'C') return link_colors['cannot'];
            })
            .attr("stroke-width", "1px");
    }
}