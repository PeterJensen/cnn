// Author: Peter Jensen
(function () {

  const preTrainedNetFile  = "cifar10_snapshot.json";
  const testBatchImageFile = "cifar10/cifar10_batch_50.png";

  var domIds = {
    startStop:  "#startStop",
    reset:      "#reset",
    samples:    "#samples",
  };
  var $Ids = {};
  
  // ------------------------
  // BEGIN CIFAR-10 SPECIFIC STUFF
  // ------------------------
  var classes = ['airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'];
  
  var testBatch          = 50;
  var numSamplesPerBatch = 1000;
  var imageDimension     = 32;
  var imageChannels      = 3;
  
  // ------------------------
  // END CIFAR-10 SPECIFIC STUFF
  // ------------------------

  // module globals
  var net;
  var running = false;
  var stepTimer = null;

  // misc utitilities
  function log(msg) {
    console.log(msg);
  }

  var Samples = (function () {
    var nextIndex = 0;
    var testBatchImageData;

    function Samples() {};
    Samples.loadBatch = loadBatch;
    Samples.next      = next;

    function loadBatch(file, done) {
      var testBatchImage = new Image();
      $(testBatchImage).on("load", function() {
        var dataCanvas = document.createElement('canvas');
        dataCanvas.width  = testBatchImage.width;
        dataCanvas.height = testBatchImage.height;
        var dataCtx = dataCanvas.getContext("2d");
        dataCtx.drawImage(testBatchImage, 0, 0);
        testBatchImageData = dataCtx.getImageData(0, 0, dataCanvas.width, dataCanvas.height);
        done();
      });
      testBatchImage.src = file;
    }

    function next() {
      var n    = testBatch*numSamplesPerBatch + nextIndex;
      var data = testBatchImageData.data;
      var vol  = new convnetjs.Vol(imageDimension, imageDimension, imageChannels, 0.0);
      var W    = imageDimension*imageDimension;
      for (var dc = 0; dc < imageChannels; dc++) {
        var i = 0;
        for (var xc = 0; xc < imageDimension; xc++) {
          for (var yc = 0; yc < imageDimension; yc++) {
            var ix = ((W * nextIndex) + i) * 4 + dc;
            vol.set(yc, xc, dc, data[ix]/255.0-0.5);
            i++;
          }
        }
      }
      if (nextIndex++ === numSamplesPerBatch) {
        nextIndex = 0;
      }
      return {vol: vol, label: labels[n]};
    }

    return Samples;
  })();

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
  
  function makeProbs(predictions, label) {
    var $div = $("<div class='probsdiv'>");
    var greenStyle = "background-color: rgb(85, 187, 85)";
    var redStyle   = "background-color: rgb(187, 85, 85)";
    for (var i = 0; i < 3; ++i) {
      var prediction = predictions[i];
      var prob100    = Math.floor(prediction.prob*100);
      var colorStyle = prediction.key === label ? greenStyle : redStyle;
      var style = "width:" + prob100 + "px;" + colorStyle;
      var $bar = $("<div class='pp' style='" + style + "'>").text(classes[prediction.key]);
      $div.append($bar);
    }
    return $div;
  }

  function sortedPredictions(vol) {
    var predictions = [];
    for (var i = 0; i < classes.length; ++i) {
      predictions.push({key: i, prob: vol.get(0, 0, i)});
    }
    predictions.sort(function (a, b) {
      return a.prob < b.prob ? 1:-1;
    });
    return predictions;
  }

  function addSample(sample, result) {
    var predictions = sortedPredictions(result);
    var $canv       = makeImage(sample.vol);
    var $probs      = makeProbs(predictions, sample.label);
    var $div        = $("<div class='testdiv'>");
    $div.append($canv, $probs);
    var $testDivs = $Ids.samples.find(">div");
    if ($testDivs.size() > 7) {
      $Ids.samples.find(">div:last").remove();
    }
    $Ids.samples.prepend($div);
  }

  function processNext() {
    log("processNext");
    var sample      = Samples.next();
    var result      = net.forward(sample.vol);
    addSample(sample, result);
  }

  function start() {
    stepTimer = setInterval(processNext, 0);
  }

  function pause() {
    clearInterval(stepTimer);
  }

  // click handlers
  function clickStartStop() {
    var $button = $Ids.startStop;
    if (running) {
      log("Pausing");
      $button.text("Start");
      pause();
    }
    else {
      log("Starting");
      $button.text("Pause");
      start();
    }
    running = !running;
  }

  function clickReset() {
    log("Resetting");
  }

  function setupClickHandlers() {
    $Ids.startStop.click(clickStartStop);
    $Ids.reset.click(clickReset);
  }

  function setJqueryIds() {
    var keys = Object.keys(domIds);
    for (var ki = 0; ki < keys.length; ++ki) {
      var key = keys[ki];
      $Ids[key] = $(domIds[key]);
    }
  }

  function main() {
    net = new convnetjs.Net();
    setJqueryIds();
    $.getJSON(preTrainedNetFile, function(json) {
      log(preTrainedNetFile + " loaded");
      net.fromJSON(json);
      Samples.loadBatch(testBatchImageFile, function () {
        log(testBatchImageFile + " loaded");
        setupClickHandlers();
      });
    });
  }

  $(main);

})();
