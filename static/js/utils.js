
function ajax_for_data(url, para_list, type) {
    let data = -1;
    $.ajaxSettings.async = false;
    $.ajax({
        url: url,
        data: para_list,
        type: type,
        success: function (result) {
            data_object.data = result.data
            data_object.label = result.label
            data_object.low_data = result.low_data
            data_object.attrs = result.attrs
        }
    });
    return data;
}


function ajax_for_get_projection(url, para_list) {
    $.ajax({
        url: url,
        data: para_list,
        type: 'POST',
        dataType: 'json',
        success: function (result) {

            let scatter_result = {embeddings: result.embeddings, label: result.label};

            let parr_data = Array();
            let attr_names = result.attrs;
            for (let i = 0; i < result.low_data.length; i++) {
                let single_obj = {}
                for (let j = 0; j < attr_names.length; j++)
                    single_obj[attr_names[j]] = result.low_data[i][j];
                parr_data.push(single_obj);
            }

            let parr_result = {'data': parr_data, 'attr': attr_names};

            scatterModel1.draw_pipeline(scatter_result);
            paraModel.drawParallelPlot(parr_result);


        }
    });

}

function ajax_for_get_projection2(url, para_list) {

    $.ajax({
        url: url,
        data: para_list,
        type: 'POST',
        dataType: 'json',
        success: function (result) {

            let scatter_result = {embeddings: result.embeddings, label: result.label};
            scatterModel1.draw_pipeline(scatter_result);

            d3.select("#project_view_scatter").selectAll('line').remove();
            // d3.select("#project_view_scatter").selectAll('line').attr("opacity", 0);

            link_object.restoreMustLink()
            link_object.restoreCannotLink()
        }
    });
}