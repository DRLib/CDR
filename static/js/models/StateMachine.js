MUST_LINK = 1
CANNOT_LINK = 0
SPREAD = 1
UN_SPREAD = 0
ACTIVE = 1
INACTIVE = 0

function StateMachine() {
    this.state='empty';
    let css_state_machine = new CssStateMachine();
    let interact_state_machine = new InteractStateMachine();
    this.change_state = function (new_state) {
        if(new_state === this.state){
            new_state = 'empty';
        }
        this.state=new_state;
        css_state_machine.change_state(new_state);
        interact_state_machine.change_state(new_state);
    };

    this.init_links = function (){
        interact_state_machine.init_links();
    }

    this.all_state_init = function () {
        this.state='empty';
        css_state_machine.all_state_init();
        interact_state_machine.all_state_init();
    };

    this.get_all_link = function () {
        return interact_state_machine.get_all_link();
    };

    this.get_link_spreads = function () {
        return interact_state_machine.get_link_spreads();
    };

    this.get_must_link = function () {
        return interact_state_machine.get_must_link();
    };

    this.get_cannot_link = function () {
        return interact_state_machine.get_cannot_link();
    };

}


function CssStateMachine() {
    let past_state='empty';
    let state2button={'add_must':'#must_button','add_cannot':'#cannot_button','delete_link':'#delete_button'};
    this.change_state = function (new_state) {
        if (new_state==='empty'){
            d3.select(".inter_button_selected").attr('class','inter_button');
            past_state = new_state;
        }
        else{
            past_state = new_state;
            d3.select(".inter_button_selected").attr('class','inter_button');
            d3.select(state2button[new_state]).attr('class','inter_button_selected');
        }
        console.log('css state',past_state)
    };
    this.all_state_init = function () {

        past_state='empty';
        d3.select(".inter_button_selected").attr('class','inter_button');
    }
}


function InteractStateMachine() {
    let past_state='empty';
    let select_node_list = [];
    let link_spread = [];
    let link_list = [];
    let must_link_list = [];
    let cannot_link_list = [];

    this.change_state = function (new_state) {
        let snapshot2DSVG = d3.select('#origin_scatter');
        // snapshot2DSVG.selectAll('circle').style("stroke",'white');
        // snapshot2DSVG = d3.select('#constrained_scatter');
        // snapshot2DSVG.selectAll('circle').style("stroke",'white');
        select_node_list = [];
        if(new_state === 'empty'){
            past_state = 'empty';
            action_query_data('#origin_scatter');
            action_query_data('#constrained_scatter');
        }
        else if(new_state==='add_must' || new_state==='add_cannot'){
            past_state = new_state;
            select_node('#origin_scatter');
            select_node('#constrained_scatter');
        }
        else if('delete_link' === new_state){
            past_state=new_state;
            delete_link('#origin_scatter');
            delete_link('#constrained_scatter');
        }
        console.log('interact state',past_state)
    };


    function select_node(svgId) {
        const snapshot2DSVG = d3.select(svgId);

        snapshot2DSVG
            .selectAll('circle')
            .on('click',function (d,i) {
                if(past_state==='add_must'||past_state==='add_cannot'){
                    if (select_node_list.length<2) {
                        if (select_node_list.length === 1 && select_node_list[0] === i)
                            return;
                        select_node_list.push(i);
                        if (select_node_list.length === 1){
                        }
                        else if (select_node_list.length === 2) {
                            send_node_list(svgId);
                            select_node_list = [];
                        }
                    }
                }
            })
    }

    function action_query_data(svgId) {

        const snapshot2DSVG = d3.select(svgId);
        snapshot2DSVG.selectAll('circle')
        .on('click',function (d, i) {
            if(past_state === 'empty')
                query_data(i);
        })
    }

    function send_node_list(svgId) {
        if (past_state === 'add_must') {
            linkModel.drawLink(svgId, select_node_list, 'must');
            select_node_list.push(MUST_LINK);
            must_link_list.push(select_node_list);
        } else if (past_state === 'add_cannot') {
            linkModel.drawLink(svgId,select_node_list, 'cannot');
            select_node_list.push(CANNOT_LINK);
            cannot_link_list.push(select_node_list);
        }
        link_list.push(select_node_list);
        // link_spread.push(SPREAD);
        link_spread.push(UN_SPREAD);
    }

    function delete_link(svgId) {

        let state = past_state;
        function search(lists, list) {
            for (let i = 0; i < lists.length; i++) {
                if (lists[i] === list[0]) {
                    if (lists[i][1] === list[1]) {
                        return i;
                    }
                }
            }
        }

        const snapshot2DSVG = d3.select(svgId);
        snapshot2DSVG
            .selectAll('line')
            .on('click', function (d, i) {
                if (state === 'delete_link') {
                    let line = d3.select(this);
                    let source = line.attr('source_');
                    let target = line.attr('target_');
                    let link_type = line.attr('link_type');
                    let index = -1;
                    if (link_type === 'must') {
                        index = search(must_link_list, [source, target]);
                        must_link_list.splice(index, 1)
                    } else {
                        index = search(cannot_link_list, [source, target]);
                        cannot_link_list.splice(index, 1)
                    }
                    index = search(link_list, [source, target]);
                    link_list.splice(index, 1);
                    d3.select(this).remove();
                }
            })
    }
    this.all_state_init = function () {

        console.log("init state!");
        past_state='empty';
        select_node_list = [];
        link_list = [];
        link_spread = [];
        must_link_list = [];
        cannot_link_list = [];
    };

    this.init_links = function () {
        link_list = [];
        link_spread = [];
        must_link_list = [];
        cannot_link_list = [];
        select_node_list = [];
    }

    this.get_link_spreads = function () {
        return link_spread;
    };

    this.get_all_link = function () {
        return link_list;
    };

    this.get_must_link = function () {
        return must_link_list;
    };

    this.get_cannot_link = function () {
        return cannot_link_list;
    }
}

