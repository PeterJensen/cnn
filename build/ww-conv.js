// Author: Peter Jensen

const receiveMessageKinds = {
  start:   "start",
  forward: "forward"
}

const sendMessageKinds = {
  log:    "log",
  result: "result"
}

var Vol = (function() {
  function Vol(sx, sy, depth) {
    this.sx = sx;
    this.sy = sy;
    this.depth = depth;
    var n = sx*sy*depth;
    this.w = zeros(n);
  }
  Vol.prototype.set = function(x, y, d, v) { 
    var ix = ((this.sx * y)+x)*this.depth+d;
    this.w[ix] = v; 
  }

  function zeros(n) {
    return new Float32Array(n);
  };
  return Vol;
})();

var layer;

function forward(V) {
  var A = new Vol(layer.out_sx |0, layer.out_sy |0, layer.out_depth |0, 0.0);
  var V_sx = V.sx |0;
  var V_sy = V.sy |0;
  var V_depth = V.depth;
  var xy_stride = layer.stride |0;

  for(var d=0;d<layer.out_depth;d++) {
    var f       = layer.filters[d];
    var f_depth = f.depth;
    var f_sx    = f.sx;
    var f_sy    = f.sy;
    var x       = -layer.pad |0;
    var y       = -layer.pad |0;
    for(var ay=0; ay<layer.out_sy; y+=xy_stride,ay++) {  // xy_stride
      x = -layer.pad |0;
      for(var ax=0; ax<layer.out_sx; x+=xy_stride,ax++) {  // xy_stride

        // convolve centered at this particular location
        var a = 0.0;
        for(var fy=0; fy < f_sy; fy++) {
          var oy = y+fy; // coordinates in the original input array coordinates
          var f_sx_X_fy = f_sx*fy;
          for(var fx=0; fx < f_sx; fx++) {
            var ox = x+fx;
            if(oy>=0 && oy<V_sy && ox>=0 && ox<V_sx) {
              var fwi   = (f_sx_X_fy+fx)*f_depth;
              var Vwi   = ((V_sx * oy)+ox)*V_depth;
              for(var fd=0;fd<f_depth;fd++) {
                // avoid function call overhead (x2) for efficiency, compromise modularity :(
                a += f.w[fwi+fd] * V.w[Vwi+fd];
              }
            }
          }
        }
        a += layer.biases.w[d];
        A.set(ax, ay, d, a);
      }
    }
  }
  return A;
}

onmessage = function (e) {
  var message = e.data;
  switch (message.kind) {
    case receiveMessageKinds.start:
      postMessage(makeMessage(sendMessageKinds.log, "start message received"));
      layer = message.payload;
      break;
    case receiveMessageKinds.forward:
      postMessage(makeMessage(sendMessageKinds.log, "forward message received"));
      var result = forward(message.payload);
      postMessage(makeMessage(sendMessageKinds.result, result));
      break;
  }
}

function makeMessage(kind, payload) {
  return {kind: kind, payload: payload};
}
