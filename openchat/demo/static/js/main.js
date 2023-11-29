function Message(arg) {
    this.text = arg.text;
    this.message_side = arg.message_side;
    this.draw = function (_this) {
        return function () {
            let $message;
            $message = $($('.message_template').clone().html());
            $message.addClass(_this.message_side).find('.text').html(_this.text);
            $('.messages').append($message);

            return setTimeout(function () {
                return $message.addClass('appeared');
            }, 0);
        };
    }(this);
    return this;
}

function getMessageText() {
    let $message_input;
    $message_input = $('.message_input');
    return $message_input.val();
}

function sendMessage(text, message_side) {
    let $messages, message;
    $('.message_input').val('');
    $messages = $('.messages');
    message = new Message({
        text: text,
        message_side: message_side
    });
    message.draw();
    $messages.animate({scrollTop: $messages.prop('scrollHeight')}, 300);
}

function onClickAsEnter(e) {
    if (e.keyCode === 13) {
        onSendButtonClicked()
    }
}

function greet() {
    setTimeout(function () {
        return sendMessage("Welcome to Open Chat's Demo !", 'left');
    }, 1000);

//    setTimeout(function () {
//        return sendMessage("Talk with AI easily using Open Chat !", 'left');
//    }, 2000);
//
//    setTimeout(function () {
//        return sendMessage("Say hello to AI.", 'left');
//    }, 3000);
}


function requestChat(imageName, messageText) {
    $.ajax({
        url: "http://127.0.0.1:8080/send/" + imageName + "/" + messageText ,
        type: "GET",
        dataType: "json",
        success: function (data) {
            console.log(data)
            return sendMessage(data["output"], 'left');
        },
        error: function (request, status, error) {
            console.log(error);
            return sendMessage('Communication failed.', 'left');
        },
        always: function(response) {
            console.log(response)
        }
    });
}
//
//function getObjectURL(file) {
//        var url = null;
//        if(window.createObjectURL!=undefined) {
//            url = window.createObjectURL(file) ;
//        }else if (window.URL!=undefined) { // mozilla(firefox)
//            url = window.URL.createObjectURL(file) ;
//        }else if (window.webkitURL!=undefined) { // webkit or chrome
//            url = window.webkitURL.createObjectURL(file) ;
//        }
//        return url ;
//    }


function onSendButtonClicked() {
    let messageText = getMessageText();
    sendMessage(messageText, 'right');
    var imageName = $("#file0")[0].files[0].name;
    return requestChat(imageName, messageText);
}

function test() {
        var fileobj = $("#file0")[0].files[0];
        var form = new FormData();
        form.append("file", fileobj);
        var out='';
        var flower='';
        $.ajax({
            type: 'POST',
            url: "image",
            data: form,
            async: false,       //同步执行
            processData: false, // 告诉jquery要传输data对象
            contentType: false, //告诉jquery不需要增加请求头对于contentType的设置
            success: function (data) {
            console.log(data)
            return sendMessage(data["output"], 'left')
        },error:function(){
                console.log("后台处理错误v2");
            }
    });

//        out.forEach(e=>{
//            flower+=`<div style="border-bottom: 1px solid #CCCCCC;line-height: 60px;font-size:16px;">${e}</div>`
//        });
//
//        document.getElementById("out").innerHTML=flower;

    }
