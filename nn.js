require('sylvester');


var exports = module.exports = {};
var fs = require('fs');
exports.input_size = 784;
exports.hidden_size = 300;
exports.weight1;
exports.weight2;

// for debuf
exports.check_NaN = function check_NaN(matrix){

    var l = matrix.cols();
    var r = matrix.rows();

    for(var i = 0; i < l; i++){
        for(var j = 0; j < r; j++){
            if(isNaN(matrix.e(j,i))){
              console.log("NaN at " + j + " " + i);
            }
        }
    }
}

// for debug
exports.save_init_weight = function save_init_weight(){
    get_init_weight();
    save_weight(weight1,weight2,'init_weight.json');
}
// for debug
exports.train_from_saved_init_weight = function train_from_saved_init_weight(){
   exports.get_saved_weight("init_weight.json",function(w1,w2){
        exports.weight1 = w1;
        exports.weight2 = w2;
        exports.train(true);
   });
}

exports.get_random_matrix = function get_random_matrix(rows,columns){
    
    var w1 = new Array(rows);
    for(var i = 0 ; i < rows; i++){
        var w = new Array(columns);
        for(var j = 0; j < columns; j++){
            w[j] = 0.1*(Math.random() - 0.5);                   
        }
        w1[i] = w;
    }
    return $M(w1);   
}

exports.get_init_weight = function get_init_weight(){
    exports.weight1 = exports.get_random_matrix(exports.hidden_size,exports.input_size + 1);
    exports.weight2 = exports.get_random_matrix(10,exports.hidden_size + 1);
}


exports.get_saved_weight = function get_saved_weight(fileName,callback){
    if(exports.weight1 == undefined){

        fs.readFile(fileName,'utf-8',function(err,data){
            if(err){
              console.log(err);
              return;
            }
            var json = eval(data);

            var w1 = json[0];
            var w2 = json[1];
            var w1_ = $M(w1);
            var w2_ = $M(w2);
            callback(w1_,w2_);
        });
      }else{
         callback(exports.weight1,exports.weight2);   
      }  
}
    

exports.sigmoid = function sigmoid(t) {
    var v=  1/(1+Math.pow(Math.E, -t));
    return v;
}

exports.elementwisely_mutiply = function elementwisely_mutiply(x,y){
    if(x.rows() != y.rows() || x.cols() != y.cols()){
        console.log("assert error in elem_by_elem_multiply()");   
    }
    var rows = x.rows();
    var columns = x.cols();

    var r = new Array(rows);
    for(var i = 0; i < rows; i++){
        var l = new Array(columns);
        for(var j = 0; j < columns; j++){
             l[j] = x.e(i+1,j+1) * y.e(i+1,j+1);
        }
        r[i] = l;
    }
    return $M(r);
}

exports.one_column = function one_column(len){
    var a = new Array(len);
    for(var i = 0; i < len; i++){
        a[i] = 1;
    }
    var m = $M([a]).transpose();
    
    return m;
    
}

exports.extract = function extract(matrix){
  var start = 2;
  var cols = matrix.cols();
  var rows = matrix.rows();

  var arr = [];
  for(var i = start; i <= rows; i++){
     arr.push(matrix.row(i).elements);
  }
  return $M(arr);
}


exports.forward_prop = function forward_prop(x){
    x.splice(0,0,1);
    x1 = $M([x]); 

    var u1 = exports.weight1.multiply(x1.transpose());
    var z = u1;
    z = z.map(exports.sigmoid);

    var z_arr = z.col(1).elements;
    z_arr.splice(0,0,1);

    var z1 = $M([z_arr]).transpose();
    var u2 = exports.weight2.multiply(z1);
    u2 = u2.map(exports.sigmoid);
    return [z,u2];
}


exports.back_prop = function back_prop(x,u,y){

    var z1 = u[0];
    var z2 = u[1];

    var eta = 0.1;
    // delta2= (z2 - y) [*] z2 [*] (1-z2)
    // [*] represent element wise multiplicaton
    var delta3 = z2.subtract(y);
    var tmp1 = exports.elementwisely_mutiply(z2,(exports.one_column(z2.rows())).subtract(z2));
    var delta2 = exports.elementwisely_mutiply(tmp1,delta3);


    // add 1 to the beginning of output vector.    
    var z_arr = z1.col(1).elements;
    z_arr.splice(0,0,1);
    var z1_ = $M([z_arr]).transpose();

    var w_delta2 = delta2.multiply(z1_.transpose());
    w_delta2 = w_delta2.map(function(t){ return 0.1 * t});

    // delta1 = (weight2 * delta2) [*] (z1 [*] (1 - z1)) 
    var tmp2 = exports.weight2.transpose().multiply(delta2);
    tmp2 = exports.extract(tmp2);
    var tmp3 = exports.elementwisely_mutiply(z1,(exports.one_column(z1.rows())).subtract(z1));
    var delta1 = exports.elementwisely_mutiply(tmp2,tmp3);
    var w_delta1 = delta1.multiply($M([x]));
    w_delta1 = w_delta1.map(function(t){ return 0.1 * t});


    exports.weight2 = exports.weight2.subtract(w_delta2);
    exports.weight1 = exports.weight1.subtract(w_delta1);

    if(exports.weight1 == null){
        console.log("assertion error :exports.weight1 == null");
    }
    if(exports.weight2 == null){
        console.log("assertion error :exports.weight1 == null");
    }
}

exports.save_weight = function save_weight(weight1,weight2,fileName){
    var rows1 = weight1.rows();
    var rows2 = weight2.rows();
    var buf = "";
    buf += '[[';
    for(var i = 1; i <= rows1; i++){
        buf += '['
        buf += weight1.row(i).elements;
        buf += '],';
    }
    buf += ']';
    buf += ',';
    buf += '[';
    for(var i = 1; i <= rows2; i++){
        buf += '[';
        buf += weight2.row(i).elements;
        buf += '],';
    }
    buf += ']]';
    fs.writeFile(fileName, buf, function (err) {
        if (err) throw err;
        console.log('It\'s saved!');
    }); 
}




exports.argmax = function argmax(arr){
  var len = arr.length;
  var max = 0;
  var index  = -1;
  for(var i = 0; i < len; i++){
     if(arr[i] > max){
        max = arr[i];
        index = i;
     }
  }
//  console.log(index);
  return index;
}

exports.predicate = function predicate(x,callback){
    exports.get_saved_weight("./weight.json",function(w1,w2){
        exports.weight1 = w1;
        exports.weight2 = w2;
        
        var output = exports.forward_prop(x);
        console.log(output[1]);
        var result = output[1].col(1).elements;
        
        callback(exports.argmax(result));
    });
    
}

exports.train = function train(skip_weight_initialize){
    if(!skip_weight_initialize)get_init_weight();
    fs.readFile('trainingSet800.json','utf-8',function(err,data){
       if(err){
         return console.log(er);
       }
       var json = eval(data);
        console.log(json.length);
       for(var i = 0; i < json.length; i++){
          if(i % 2000 == 0){
             console.log(i + " training done");
          }
          var input = json[i].input;
          var output = json[i].output;
          var x = input;

          var y = output;
          var u = exports.forward_prop(x);
          exports.back_prop(x,u,$M([y]).transpose());
       }
       exports.save_weight(exports.weight1,exports.weight2,"weight.json");
       console.log("train done");
    });

}

exports.test = function test(){
    exports.get_saved_weight("./weight.json",function(w1,w2){
        exports.weight1 = w1;
        exports.weight2 = w2;

        fs.readFile('testSet2000.json','utf-8',function(err,data){
           if(err){
             return console.log(err);
           }
           var percentage = 0;
           var json = eval(data);
           for(var i = 0; i < json.length; i++){
              var input = json[i].input;
              var output = json[i].output;
              var x = input;
              var y = output;
              var u = exports.forward_prop(x);
              var predicted = exports.argmax(u[1].col(1).elements);
              output = exports.argmax(output);
              if(predicted == output){
                 percentage += 1;
              }
           }
           console.log("percentage: " + percentage/2000);
        });
    });
}



exports.test();
//exports.train_from_saved_init_weight();
export.train(false);