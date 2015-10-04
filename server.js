http = require('http');
fs = require('fs');
nn = require('./nn.js');


server = http.createServer( function(req, res) {

    if (req.method == 'POST') {
        console.log("POST");
        var body = '';
        req.on('data', function (data) {
            body += data;
           // console.log("Partial body: " + body);
        });

        req.ona('end', function () {
            var input = JSON.parse(body).input;
            for(var i = 0 ; i < input.length; i++){
                input[i] = input[i]/255;
            }
            console.log(input);
            var output = nn.predicate(input,function(result){
              res.writeHead(200, {'Content-Type': 'text/html'});
              res.end("'" + result + "'");
            });
        });
    }
    else{
        var html = fs.readFileSync('index.html');
        res.writeHead(200, {'Content-Type': 'text/html'});
        res.end(html);
    }

});

port = 3000;
host = '<IPADDRESS>';
server.listen(port, host);
console.log('Listening at http://' + host + ':' + port);
