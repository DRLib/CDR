Array.prototype.unique = function (a) {
    return function () {
        return this.filter(a)
    }
}(function (a, b, c) {
    return c.indexOf(a, b + 1) < 0
});

function NewScatterModel(svgId) {
    this.svg = d3.select(svgId);
    this.width = this.svg.node().parentNode.clientWidth;
    this.height = this.svg.node().parentNode.clientHeight;
    this.svg.attr("width", this.width).attr("height", this.height);
    this.margin = {top: 20, right: 20, bottom: 20, left: 20};

    this.canvas = this.svg.append('g');
    this.links_canvas = this.canvas.append('g').attr("id", "links_canvas");
    this.points_canvas = this.canvas.append('g');
    this.pictures_canvas = this.canvas.append('g');

    var tooltip = d3.select("body")
        .append("div")
        .attr("opacity", 0)
        .attr("class", "tooltip");
    var text_tooltip = tooltip.append('div')
    var image_tooltip = tooltip.append('img')

    this.max_sample_num = 30
    this.sample_num = 10
    this.img_dir = "static/images/"
    this.lasso_ids = []

    this.draw_pipeline = function (result) {
        this.result = result
        this.process(result)
        this.pictures_canvas.selectAll('image').remove()
        this.points_canvas.selectAll('circle').remove()
        this.links_canvas.selectAll("line").remove()
        if (project_object.picture_state == 0) {
            this.draw_points()
        } else if (project_object.picture_state == 1) {
            this.draw_pictures()
        } else {
            console.log('error')
        }
        linkModel.listen_link()
        this.call_mode()
    }

    this.process = function (result) {
        this.data = result.embeddings
        this.label = result.label
        this.label_map()
        const max = {};
        const min = {};
        max.x = d3.max(this.data, d => d[0]);
        max.y = d3.max(this.data, d => d[1]);
        min.x = d3.min(this.data, d => d[0]);
        min.y = d3.min(this.data, d => d[1]);
        this.xScale = d3
            .scaleLinear()
            .domain([min.x, max.x])
            .range([this.margin.left, this.width - this.margin.right]);
        this.yScale = d3
            .scaleLinear()
            .domain([-max.y, -min.y])
            .range([this.margin.top, this.height - this.margin.bottom]);
    }

    this.draw_points = function () {
        let lasso_ids = this.lasso_ids
        let self_obj = this;
        this.items = this.points_canvas.selectAll("circle")
            .data(this.data)
            .enter()
            .append("circle")
            .attr("cx", d => this.xScale(d[0]))
            .attr("cy", d => this.yScale(-d[1]))
            .attr("r", 4)
            .attr("id", (d, i) => "circle_" + i)
            .attr("cls", (d, i) => this.label[i])
            .attr("fill", (d, i) => node_colors['normal'])
            .on('mouseover', function (d, i) {

                tooltip
                    .style("display", "block")
                    .style("left", (d3.event.pageX + 5) + "px")
                    .style("top", (d3.event.pageY - 35) + "px")
                    .style("opacity", 1);

                text_tooltip.html("ID:" + i + " ")

                if (para_object.selected_dataset_type == 'image') {
                    image_tooltip
                        .attr('src', self_obj.generate_path(i))
                        .attr("width", "100")
                        .attr("height", "100")
                        .style("display", "block")
                } else {
                    image_tooltip.style("display", "none")
                }

                if (lasso_ids.indexOf(i) > -1) return true;
                d3.select(this)
                    .attr("fill", node_colors['hover'])
                    .attr("r", 7)
                paraModel.highlightSingleLine(i);
            })
            .on('mouseout', function (d, i) {

                tooltip.style("display", "none")

                if (lasso_ids.indexOf(i) > -1) return true;
                d3.select(this)
                    .attr("fill", node_colors['normal'])
                    .attr("r", 4)

                paraModel.notHighlightSingleLine(i);

            })

        if (lasso_ids.length > 0) {
            this.points_canvas.selectAll("circle")
                .filter(function (d, i) {
                    if (lasso_ids.indexOf(i) > -1) return true;
                    else return false
                })
                .attr("fill", node_colors['lasso'])
            paraModel.highlightLines(lasso_ids)
        }
    }

    this.draw_pictures = function () {
        let xScale = this.xScale
        let yScale = this.yScale
        let lasso_ids = this.lasso_ids
        link_object.getLinkID()

        let picture_to_draw_ids = sample_images(this.label, this.sample_num).concat(lasso_ids).concat(link_object.link_ids)
        let self_obj = this;
        this.items = this.pictures_canvas.selectAll("image")
            .data(this.data)
            .enter()
            .append("image")
            .attr("x", d => this.xScale(d[0]) - project_object.picture_width / 2)
            .attr("y", d => this.yScale(-d[1]) - project_object.picture_height / 2)
            .attr("id", (d, i) => "images_" + i)
            .attr('opacity', 0.8)
            .style('display', function (d, i) {

                if (picture_to_draw_ids.indexOf(i) > -1) return "block";
                else return "none";
            })
            .attr('xlink:href', function (d, i) {
                if (picture_to_draw_ids.indexOf(i) > -1) return self_obj.generate_path(i);
            })
            .attr("width", project_object.picture_width)
            .attr("height", project_object.picture_height)
            .on('mouseover', function (d, i) {
                tooltip
                    .style("display", "block")
                    .style("left", (d3.event.pageX + 5) + "px")
                    .style("top", (d3.event.pageY - 35) + "px")
                    .style("opacity", 1.0);
                text_tooltip.html("ID:" + i + " ")
                image_tooltip
                    .attr('src', self_obj.generate_path(i))
                    .attr("width", "120")
                    .attr("height", "120")
                    .style("display", "block")
                if (lasso_ids.indexOf(i) > -1) return true;
                let x = xScale(d[0]) - project_object.picture_width / 2 - 1
                let y = yScale(-d[1]) - project_object.picture_height / 2 - 1
                d3.select(this)
                    .attr("x", x)
                    .attr("y", y)
                    .attr("width", project_object.picture_width + 2)
                    .attr("height", project_object.picture_height + 2)

                paraModel.highlightSingleLine(i);
            })
            .on('mouseout', function (d, i) {
                tooltip.style("display", "none")
                if (lasso_ids.indexOf(i) > -1) return true;
                let x = xScale(d[0]) - project_object.picture_width / 2
                let y = yScale(-d[1]) - project_object.picture_height / 2
                d3.select(this)
                    .attr("x", x)
                    .attr("y", y)
                    .attr("width", project_object.picture_width)
                    .attr("height", project_object.picture_height)
                paraModel.notHighlightSingleLine(i);

            })

        if (lasso_ids.length > 0) {
            this.pictures_canvas.selectAll("image")
                .filter(function (d, i) {
                    if (lasso_ids.indexOf(i) > -1) return true;
                    else return false
                })
                .attr("opacity", 1)
            paraModel.highlightLines(lasso_ids)
        }
    }

    this.call_mode = function () {
        if (project_object.mustlink_state || project_object.cannotlink_state) {
            console.log("link")
            this.call_link_mode()
        } else {
            console.log("lasso")
            this.call_lasso_mode()
        }
    }

    this.call_link_mode = function () {

        this.remove_lasso()
        this.svg.on(".drag", null);
    }

    this.call_lasso_mode = function () {
        let svg = this.svg
        svg.on(".zoom", null)

        var lasso = d3.lasso()
            .closePathSelect(true)
            .closePathDistance(100)
            .targetArea(svg)

        if (project_object.picture_state == 0) {
            lasso = this.call_points_lasso_mode(svg, lasso)
        } else {
            lasso = this.call_pictures_lasso_mode(svg, lasso)
        }
        svg.call(lasso)
    }

    this.call_points_lasso_mode = function (svg, lasso) {
        let lasso_ids = this.lasso_ids
        var lasso_start = function () {
            lasso.items()
                .attr("r", 4) // reset size
                .attr("fill", node_colors['not_lasso'])

            lasso_ids.splice(0, lasso_ids.length)
            paraModel.notHighlightLines();
        };

        var lasso_draw = function () {
            // Style the possible dots
            lasso.possibleItems()
                .attr("fill", node_colors['lasso'])

            // Style the not possible dot
            lasso.notPossibleItems()
                .attr("fill", node_colors['not_lasso'])
        };

        var lasso_end = function () {
            // Reset the color of all dots
            lasso.items()
                .attr("fill", node_colors['normal'])

            // Style the selected dots
            lasso.selectedItems()
                .attr("fill", node_colors['lasso'])

            let selected = lasso.selectedItems()._groups[0]
            for (let i = 0; i < selected.length; i++)
                lasso_ids.push(parseInt(selected[i].id.slice(7)))
            console.log(lasso_ids);
            // Reset the style of the not selected dots
            paraModel.highlightLines(lasso_ids)
        };


        lasso.items(svg.selectAll("circle"))
            .on("start", lasso_start)
            .on("draw", lasso_draw)
            .on("end", lasso_end);

        return lasso
    }

    this.call_pictures_lasso_mode = function (svg, lasso) {
        let lasso_ids = this.lasso_ids
        var lasso_start = function () {
            lasso.items()
                .classed("image_not_possible", true)
                .classed("image_possible", false)
            console.log(lasso_ids)

            lasso_ids.splice(0, lasso_ids.length)
            paraModel.notHighlightLines();
        };

        var lasso_draw = function () {
            // Style the possible dots
            lasso.possibleItems()
                .classed("image_not_possible", false)
                .classed("image_possible", true)

            // Style the not possible dot
            lasso.notPossibleItems()
                .classed("image_not_possible", true)
                .classed("image_possible", false)
        };

        var lasso_end = function () {
            // Style the selected dots
            lasso.selectedItems()
                .classed("image_selected", true)
                .classed("image_possible", false)

            let selected = lasso.selectedItems()._groups[0]
            for (let i = 0; i < selected.length; i++)
                lasso_ids.push(parseInt(selected[i].id.slice(7)))
            // Reset the style of the not selected dots
            lasso.notSelectedItems()
                .classed("image_selected", false)
                .classed("image_possible", false)
            console.log(lasso_ids)
            paraModel.highlightLines(lasso_ids)
        };

        lasso.items(svg.selectAll("image"))
            .on("start", lasso_start)
            .on("draw", lasso_draw)
            .on("end", lasso_end);

        return lasso
    }

    this.remove_lasso = function () {
        let svg = this.svg
        if (project_object.picture_state == 0) {
            svg.selectAll('circle')
                .attr("r", 4)
                .classed("point_not_possible", false)
                .classed("point_possible", false)
                .classed("point_selected", false)
        } else {
            svg.selectAll('image')
                .classed("image_selected", false)
                .classed("image_possible", false)
                .classed("image_not_possible", true)
        }
    }

    this.generate_path = function (idx) {
        let src_dir = this.img_dir + para_object.selected_dataset_name;
        return src_dir + "/" + idx + ".jpg"

    }

    this.label_map = function () {
        let unique_label = this.label.unique();
        this.label_dict = {};
        for (let i = 0; i < unique_label.length; i++) {
            this.label_dict[unique_label[i]] = i;
        }
    }
}

function sample_images(labels, sample_num) {
    let n_samples = labels.length;
    let sampled_indices = [];
    let label_count = {};
    let unique_label = labels.unique();
    for (let i = 0; i < unique_label.length; i++)
        label_count[unique_label[i]] = 0

    let indices = generateArray(0, n_samples-1);
    indices = shuffleSelf(indices, n_samples);
    // console.log(indices)

    for (let i = 0; i < n_samples; i++) {
        let idx = indices[i]
        let cur_label = labels[idx];
        if (label_count[cur_label] <= sample_num) {
            label_count[cur_label] += 1
            sampled_indices.push(idx)
        }
    }
    return sampled_indices
}

function generateArray(start, end) {
    return Array.from(new Array(end + 1).keys()).slice(start)
}

function shuffleSelf(array, size) {
    var index = -1,
        length = array.length,
        lastIndex = length - 1;

    size = size === undefined ? length : size;
    while (++index < size) {
        var rand = index + Math.floor(Math.random() * (lastIndex - index + 1))
        value = array[rand];

        array[rand] = array[index];

        array[index] = value;
    }
    array.length = size;
    return array;
}