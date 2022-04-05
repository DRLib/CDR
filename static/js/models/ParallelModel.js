function ParallelModel(svgId) {
    const parallelSVG = d3.select(svgId);
    const width = parallelSVG.node().parentNode.clientWidth;
    const height = parallelSVG.node().parentNode.clientHeight;
    parallelSVG.attr("width", width).attr("height", height);
    let margin = {top: 30, right: 40, bottom: 15, left: 40};

    this.drawParallelPlot = function(result){
        parallelSVG.selectAll('g').remove();
        this.line_g1 = parallelSVG.append('g');
        this.line_g2 = parallelSVG.append('g');
        this.line_g3 = parallelSVG.append('g');
        this.axis_g = parallelSVG.append('g');
        this.text_g = parallelSVG.append('g');
        let attr = result.attr
        let data = result.data
        this.xScale=d3.scaleLinear()
            .domain([0, attr.length-1])
            .range([margin.left, width - margin.right]);

        if (para_object.selected_dataset_name === "Wifi") {
            this.yScales = d3.zip(...(data.map((item) => d3.permute(item, attr)))).map((subject) => {
                return d3.scaleLinear()
                            .domain([-100, -10])
                            .range([margin.top, height - margin.bottom]);
            });
        } else {
            this.yScales = d3.zip(...(data.map((item) => d3.permute(item, attr)))).map((subject) => {
                return d3.scaleLinear()
                            .domain([d3.min(subject), d3.max(subject)])
                            .range([margin.top, height - margin.bottom]);
            });
        }

        // this.yScales = d3.zip(...(data.map((item) => d3.permute(item, attr)))).map((subject) => {
        //     return d3.scaleLinear()
        //                 .domain([d3.min(subject), d3.max(subject)])
        //                 .range([margin.top, height - margin.bottom]);
        // });

        this.renderLines(data,attr);
        this.renderAxis();
        this.renderText(attr)
    }

    this.renderAxis = function(){
        let axis_g = this.axis_g;
        xScale = this.xScale
        yScales = this.yScales
        yScales.forEach((scale, index) => {
            axis_g
                 .append('g')
                 .attr('transform', 'translate(' + xScale(index) + ',0)' )
                 .call(d3.axisLeft(scale).ticks(5));
        });
        axis_g.selectAll("text")
        .attr("font-size",12)

        axis_g.selectAll("text")
        .clone(true)
        .lower()
        .attr('fill', 'none')
        .attr('stroke-width', 5)
        .attr('stroke-linejoin', 'round')
        .attr('stroke', 'white')
    }

    this.renderLines = function(data,attr){
        let xScale = this.xScale
        let yScales = this.yScales
        function generatePoints(d) {
            return d3.permute(d, attr).map((item, index) => {
                return [
                    xScale(index),
                    yScales[index](item)
                ];
            });
        }
        
        let line_g1 = this.line_g1
                        .selectAll('.line')
                        .data(data);
        
        const linesEnter1 = line_g1.enter()
                        .append('g')

        linesEnter1.append('path')
                        .attr('stroke', node_colors['lasso_line'])
                        .attr('opacity',0.1)
                        .attr('stroke-width', 2)
                        .attr('fill', 'none')
                        .attr('d', (d) => d3.line()(generatePoints(d)))

        let line_g2 = this.line_g2
        .selectAll('.line')
        .data(data);

        const linesEnter2 = line_g2.enter()
                        .append('g')

        linesEnter2.append('path')
                        .attr('stroke', node_colors['lasso_line'])
                        .attr('opacity',0)
                        .attr('stroke-width', 2)
                        .attr('fill', 'none')
                        .attr('d', (d) => d3.line()(generatePoints(d)))

         let line_g3 = this.line_g3
        .selectAll('.line')
        .data(data);

        const linesEnter3 = line_g3.enter()
                        .append('g')

        linesEnter3.append('path')
                        .attr('stroke', node_colors['lasso_line'])
                        .attr('opacity',0)
                        .attr('stroke-width', 2)
                        .attr('fill', 'none')
                        .attr('d', (d) => d3.line()(generatePoints(d)))
    }
    
    this.renderText = function(attr){
        let text_g = this.text_g
        .selectAll('text')
        .data(attr)
        .enter()
        .append('g')
        .append('text')
        .attr('dx','1em')
        .attr('transform', (d,i)=>'translate(' + (xScale(i)-20) + ',10)' )
        .style("text-anchor", "middle")
        .text((d,i)=>d)
    }

    this.highlightSingleLine = function(id){
        this.line_g3.selectAll('path')
        .filter(function(d,i){
            if(id==i)return true;
            else return false;
        })
        .attr('stroke', node_colors['hover'])
        .attr('stroke-width', 3)
        .attr('opacity',0.5)
    }

    this.notHighlightSingleLine = function(id){
        this.line_g3.selectAll('path')
        .filter(function(d,i){
            if(id==i)return true;
            else return false;
        })
        .attr('stroke', node_colors['lasso_line'])
        .attr('stroke-width', 2)
        .attr('opacity', 0)
    }

    this.highlightLines = function(ids){

        if(ids.length==0) return
        this.line_g2.selectAll('path').attr('opacity',0)
        this.line_g2.selectAll('path')
        .filter(function(d,i){
            if(ids.indexOf(i)>-1)return true;
            else return false;
        })
        .attr('stroke', node_colors['lasso'])
        .attr('stroke-width', 2)
        .attr('opacity',0.5)
    }

    this.notHighlightLines = function(){
         this.line_g2.selectAll('path').attr('opacity',0)
        this.line_g2.selectAll('path')
        .attr('stroke', node_colors['lasso_line'])
        .attr('stroke-width', 2)
        .attr('opacity', 0)
    }
    
}