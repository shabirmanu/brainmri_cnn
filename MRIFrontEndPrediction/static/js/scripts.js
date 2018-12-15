jsPlumb.ready(function() {

    jsPlumb.setContainer($('.flowchart'))
    var common = {
        anchor:"Continuous",
        endpoint:["Rectangle", { width:20, height:20 }],
        paintStyle: {fill:"gray", stroke:"gray", strokeWidth:3}
    };
    var i = 0;
    var source_ids = [];
    var target_ids = [];

    source = [];
    target = [];

    $('.vo-item').each(function() {
       source.push(this.id)
    });

    $('.task-item').each(function() {
       target.push(this.id)
    });

    console.log(source)


     jsPlumb.makeSource(source, {
        connector: 'StateMachine'
    }, common);
    jsPlumb.makeTarget(target, {
        connector: 'StateMachine',
        allowLoopback: true,
        maxConnection:3,
        isTarget:true,
        overlays:[
            ["Arrow" , { width:12, length:12, location:0.67 }]
        ]


    }, common);


});

$(function() {
    $('.spinner').hide();
    $('a.deploy').bind('click', function() {
        $(this).text('Deploying')
        url = $(this).attr('href');
         URLArr = url.split('?')
         mainURL = URLArr[0]
         task = URLArr[1].split('=')[1]

        console.log(mainURL)
         id = $(this).attr('id');
         $('#spinner-'+id).show();
        $.ajax({
              type: "GET",
              dataType: 'json',
              data: {'id': id, 'task':task},
              crossDomain:true,
              url: url
        }).done(function (data) {
            $('#spinner-'+id).hide();
            console.log(data.success)
            if(data.success) {
                $('#'+id).text('Deployed');
                $('#'+id).addClass('btn-danger')
                $('#'+id).attr('href','#');
            }
        }).fail(function(data) {
        });


        // console.log(url)
        // console.log(id)
        //
        //
        // $.get(url, {id: id}, function(data) {
        // $("#result").text(data.result);
        // });

        return false;
    });
  });

