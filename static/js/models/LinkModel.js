function LinkModel() {

    const svg = d3.select("#project_view_scatter");
    this.listen_link = function(){
        const svg = d3.select("#project_view_scatter");
        let select_node_list = [];
        let model = this
        function add_link_data(item){
            svg.selectAll(item)
            .on('click',function (d,i) {
                console.log(d)
                if(project_object.mustlink_state || project_object.cannotlink_state){
                    if (select_node_list.length<2) {
                        if (select_node_list.length === 1 && select_node_list[0] === i)
                            return;
                        select_node_list.push(i);
                        if (select_node_list.length === 2) {
                            model.send_node_list(select_node_list);
                            select_node_list = [];
                        }
                    }
                }
            })
        }
        if(project_object.picture_state==0){
            console.log("circle listen")
            add_link_data('circle')
        }else{
            console.log("image listen")
            add_link_data('image')
        }
    }

    this.send_node_list = function(select_node_list){
        if (project_object.mustlink_state) {
            link_object.addMustLink(select_node_list)
        } else if (project_object.cannotlink_state) {
            link_object.addCannotLink(select_node_list)
        }
    }

    this.draw_link_pipeline = function(select_node_list,link_type){
        if(project_object.picture_state==0){
            let arr = this.get_points_position(select_node_list,link_type)
            this.draw_link(arr,select_node_list,link_type)
        }else if(project_object.picture_state==1){
            let arr = this.get_picture_position(select_node_list,link_type)
            this.draw_link(arr,select_node_list,link_type)
        }
    }
    this.get_points_position = function(select_node_list){
        let nodes = svg.selectAll('circle');
        let source = nodes._groups[0][select_node_list[0]];
        let target = nodes._groups[0][select_node_list[1]];
        let x1 = source.cx.animVal['value'];
        let y1 = source.cy.animVal['value'];
        let x2 = target.cx.animVal['value'];
        let y2 = target.cy.animVal['value'];
        return [x1,y1,x2,y2]
    }
    this.get_picture_position = function(select_node_list){
        let nodes = svg.selectAll('image');
        let source = nodes._groups[0][select_node_list[0]];
        let target = nodes._groups[0][select_node_list[1]];
        let x1 = source.x.animVal['value']+project_object.picture_width/2;
        let y1 = source.y.animVal['value']+project_object.picture_height/2;
        let x2 = target.x.animVal['value']+project_object.picture_width/2;
        let y2 = target.y.animVal['value']+project_object.picture_height/2;;
        return [x1,y1,x2,y2]
    }
    this.draw_node_border = function(){
        let mustlink_id =link_object.get_mustlink_id()
        let cannotlink_id = link_object.get_cannotlink_id()
        d3.selectAll("circle").attr("stroke","")
        mustlink_id.forEach(id => {
            d3.select("#circle_"+id).attr("stroke",link_colors['must'])
        });
        cannotlink_id.forEach(id => {
            d3.select("#circle_"+id).attr("stroke",link_colors['cannot'])
        });
    }
    this.draw_link = function(arr,select_node_list,link_type) {
        x1 = arr[0],y1 = arr[1],x2 = arr[2],y2 = arr[3]
        let links_canvas = d3.select('#links_canvas');
        this.draw_node_border()

        links_canvas.append("line")
            .attr("x1", x1)
            .attr("y1", y1)
            .attr("x2", x2)
            .attr("y2", y2)
            .attr("id","line_"+select_node_list[0]+"_"+select_node_list[1])
            .attr('source_',select_node_list[0])
            .attr('target_',select_node_list[1])
            .attr('link_type',link_type)
            .attr("stroke", function (d) {
                if(link_type==='must')return link_colors['must'];
                else if(link_type==='cannot') return link_colors['cannot'];
            })
            .attr("stroke-width", "2px")
            .attr("opacity",function(d,i){
                if(select_node_list[2]==true)return 0.5;
                else if(select_node_list[2]==false)return 0;
            });
    };

    this.delete_link = function(link) {
        const svg = d3.select("#project_view_scatter");
        this.draw_node_border()
        svg.select("#line_"+link.head+"_"+link.tail).remove();
    }

    this.active_link = function(link){
        const svg = d3.select("#project_view_scatter");
        this.draw_node_border()
        console.log(link);
        let type = link.index[0] === "M"? "must":"cannot";
        this.draw_link_pipeline([link.head, link.tail], type);
        svg.select("#line_"+link.head+"_"+link.tail).attr("opacity", 0.5);
    }

    this.not_active_link = function(link){
        const svg = d3.select("#project_view_scatter");
        this.draw_node_border()
        svg.select("#line_"+link.head+"_"+link.tail).attr("opacity", 0);
    }

    this.highlight_link = function(link){
        const svg = d3.select("#project_view_scatter");
        svg.select("#line_"+link.head+"_"+link.tail).attr("stroke-width", "3px");
    }

    this.not_highlight_link = function(link){
        const svg = d3.select("#project_view_scatter");
        svg.select("#line_"+link.head+"_"+link.tail).attr("stroke-width", "2px");
    }
}