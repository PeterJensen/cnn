// Author: Peter Jensen

(function () {

  const domIds = {
    video:       "video",
    canvasBig:   "#canvasBig",
    canvasSmall: "#canvasSmall",
    startStop:   "#startStop",
    result:      "#result"
  }
  var $domIds = {};

  const videoDim = {
    width:        300,
    height:       300
  }
  const canvasDim = {
    width:  512,
    height: 512
  }
  const canvasSmallDim = {
    width:  32,
    height: 32
  }
  const imageChannels = 3;
  const preTrainedNetFile  = "cifar10_snapshot.json";
  const classes            = ['airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'];

  // these numbers are computed after the video stream is initialized
  var videoCrop = {
    topLeft:      {x: 0, y: 0},
    width:        480,
    height:       480,
    actualWidth:  640,
    actualHeight: 480
  }

  var ctxBig;
  var ctxSmall;
  var timer = null;
  var net;

  function log(msg) {
    console.log(msg);
  }

  function getData(ctx) {
    var width  = canvasSmallDim.width;
    var height = canvasSmallDim.height;
    var imageData = ctx.getImageData(0, 0, width, height);
    var data = imageData.data;
    var vol  = new convnetjs.Vol(width, height, imageChannels, 0.0);
    for (var dc = 0; dc < imageChannels; dc++) {
      var i = 0;
      for (var xc = 0; xc < width; xc++) {
        for (var yc = 0; yc < height; yc++) {
          var ix = 4*i + dc;
          vol.set(yc, xc, dc, data[ix]/255.0-0.5);
          i++;
        }
      }
    }
    return vol;
  }

  function makeImage(vol, scale, grads) {
    var scale = (typeof scale === "undefined") ? 2 : scale;
    var grads = (typeof grads === "undefined") ? false : grads;

    // get max and min activation to scale the maps automatically
    var w  = grads ? vol.dw : vol.w;
    var mm = cnnutil.maxmin(w);

    var $canv = $("<canvas>");
    var W = vol.sx * scale;
    var H = vol.sy * scale;
    $canv.attr("width", W);
    $canv.attr("height", H);
    var ctx = $canv[0].getContext('2d');
    var g = ctx.createImageData(W, H);
    for (var d = 0; d < 3; d++) {
      for (var x = 0; x < vol.sx; x++) {
        for (var y = 0; y < vol.sy; y++) {
          var dval;
          if (grads) {
            dval = Math.floor((vol.get_grad(x,y,d)-mm.minv)/mm.dv*255);
          }
          else {
            dval = Math.floor((vol.get(x,y,d)-mm.minv)/mm.dv*255);  
          }
          for (var dx = 0; dx < scale; dx++) {
            for (var dy = 0; dy < scale; dy++) {
              var pp = ((W * (y*scale + dy)) + (dx + x*scale)) * 4;
              g.data[pp + d] = dval;
              if (d === 0) {
                g.data[pp+3] = 255; // alpha channel
              }
            }
          }
        }
      }
    }
    ctx.putImageData(g, 0, 0);
    return $canv;
  }

  function makeProbs(vol) {
    var $div = $("<div>");
    for (var i = 0; i < classes.length; ++i) {
      var prediction = vol.get(0, 0, i);
      var prob100    = Math.floor(prediction*100);
      var style      = "width:" + prob100 + "%";
      var $bar = $("<div class='cc-bar'" + " style='" + style + "'>").text(classes[i]);
      $div.append($bar);
    }
    return $div;
  }
  
  function sortedPredictions(vol) {
    var predictions = [];
    for (var i = 0; i < classes.length; ++i) {
      predictions.push({key: i, prediction: vol.get(0, 0, i)})
    }
    predictions.sort(function (a, b) {
      return a.prediction < b.prediction ? 1 : -1;
    });
    return predictions;
  }

  function makePrediction(vol) {
    var sorted = sortedPredictions(vol);
    var $div = $("<div class='cc-prediction'>");
    $div.text("It's a " + classes[sorted[0].key]);
    return $div;
  }

  function makeSample(sampleVol, result) {
    var $canv       = makeImage(sampleVol, 4);
    var $probs      = makeProbs(result);
    var $prediction = makePrediction(result);
    var $div        = $("<div>");
    $div.append($canv, $probs, $prediction);
    return $div;
  }

  function update() {
    var video = $domIds.video[0];
    ctxBig.drawImage(
      video,
      videoCrop.topLeft.x, videoCrop.topLeft.y, videoCrop.width, videoCrop.height,
      0, 0, canvasDim.width, canvasDim.height);
    ctxSmall.drawImage(
      video,
      videoCrop.topLeft.x, videoCrop.topLeft.y, videoCrop.width, videoCrop.height,
      0, 0, canvasSmallDim.width, canvasSmall.height);
    var vol = getData(ctxSmall);
    var result = net.forward(vol);
    var $sample = makeSample(vol, result);
    $domIds.result.empty();
    $domIds.result.append($sample);
  }

  function clickStartStop() {
    if (timer !== null) {
      clearInterval(timer);
      timer = null;
      $domIds.startStop.text("Start");
    }
    else {
      timer = setInterval(update, 0);
      $domIds.startStop.text("Stop");
    }
  }

  function setJqueryIds() {
    var keys = Object.keys(domIds);
    for (var ki = 0; ki < keys.length; ++ki) {
      var key = keys[ki];
      $domIds[key] = $(domIds[key]);
    }
  }

  function setDrawDims() {
    var video = $domIds.video[0];
    videoCrop.actualWidth  = video.videoWidth;
    videoCrop.actualHeight = video.videoHeight;
    var arSrc = video.videoWidth/video.videoHeight;
    var arDst = canvasDim.width/canvasDim.height;
    if (arSrc > arDst) {
      // Crop left and right
      videoCrop.width     = arDst*video.videoHeight;
      videoCrop.height    = video.videoHeight;
      videoCrop.topLeft.x = (video.videoWidth - videoCrop.width)/2;
      videoCrop.topLeft.y = 0;
    }
    else {
      // Crop top and bottom
      videoCrop.width  = video.videoWidth;
      videoCrop.height = video.videoWidth/arDst;
      videoCrop.topLeft.x = 0;
      videoCrop.topLeft.y = (video.videoHeight - videoCrop.height)/2;
    }
  }

  function setDomAttr() {
    $domIds.video.attr("width", videoDim.width);
    $domIds.video.attr("height", videoDim.height);
    $domIds.video.attr("autoplay", true);
    $domIds.canvasBig.attr("width", canvasDim.width);
    $domIds.canvasBig.attr("height", canvasDim.height);
    $domIds.canvasSmall.attr("width", canvasSmallDim.width);
    $domIds.canvasSmall.attr("height", canvasSmallDim.height);
    $domIds.startStop.on("click", clickStartStop);
  }

  function main() {
    setJqueryIds();
    net      = new convnetjs.Net();
    ctxBig   = $domIds.canvasBig[0].getContext("2d");
    ctxSmall = $domIds.canvasSmall[0].getContext("2d");
    navigator.mediaDevices.getUserMedia({video: true})
      .then(function (stream) {
        var video = $domIds.video[0];
        video.srcObject = stream;
        video.onloadedmetadata = function() {
          setDrawDims();
        };
      })
      .catch(function (error) {
        log("ERROR: " + error.message);
      });
    $.getJSON(preTrainedNetFile, function(json) {
      log(preTrainedNetFile + " loaded");
      net.fromJSON(json);
      setDomAttr();
    });
  }

  $(main);

})();
