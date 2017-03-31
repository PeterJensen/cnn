// Author: Peter Jensen
(function () {

  const preTrainedNetFile  = "cifar10_snapshot.json";
  const testBatchImageFile = "cifar10/cifar10_batch_50.png";
  const samplesCount       = 16;

  const domIds = {
    startStop:        "#startStop",
    reset:            "#reset",
    samples:          "#samples",
    testImageCount:   "#testImageCount",
    testTotalTime:    "#testTotalTime",
    testTimePerImage: "#testTimePerImage",
    accuracy:         "#accuracy"
  };
  var $domIds = {}; // jQuery Ids of referenced DOM elements.  Initialized after page load
  
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
  var running   = false;
  var stepTimer = null;
  var timing;
  var stats;

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
    Samples.reset     = reset;

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

    function reset() {
      nextIndex = 0;
    }

    return Samples;
  })();

  var Timing = (function () {

    function Timing() {
      this.count = 0;
      this.total = 0;
    }

    Timing.prototype.record = function(fct) {
      var startTime = Date.now();
      fct();
      var stopTime = Date.now();
      this.count++;
      this.total += (stopTime - startTime);
    }
    Timing.prototype.reset = function() {
      this.count = 0;
      this.total = 0;
    }

    return Timing;

  })();

  var Stats = (function () {

    function Stats() {
      this.count = 0;
      this.successCount = 0;
    }
    Stats.prototype.record = function (success) {
      this.count++;
      if (success) {
        this.successCount++;
      }
    }
    Stats.prototype.reset = function () {
      this.count = 0;
      this.successCount = 0;
    }

    return Stats;

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
    var $testDivs = $domIds.samples.find(">div");
    if ($testDivs.size() > samplesCount - 1) {
      $domIds.samples.find(">div:last").remove();
    }
    $domIds.samples.prepend($div);
    stats.record(predictions[0].key === sample.label);
  }

  function updateStats() {
    $domIds.testImageCount.text(timing.count);
    $domIds.testTotalTime.text(timing.total + "ms");
    if (timing.count > 0) {
      $domIds.testTimePerImage.text(Math.round(timing.total/timing.count) + "ms");
    }
    else {
      $domIds.testTimePerImage.text("N/A");
    }
    if (stats.count > 0) {
      $domIds.accuracy.text((100*stats.successCount/stats.count).toFixed(1) + "%");
    }
    else {
      $domIds.accuracy.text("N/A");
    }
  }

  function processNext() {
    var sample = Samples.next();
    var result;
    timing.record(function () {
      result = net.forward(sample.vol);
    });
    addSample(sample, result);
    updateStats();
  }

  function start() {
    stepTimer = setInterval(processNext, 0);
  }

  function pause() {
    clearInterval(stepTimer);
  }

  // click handlers
  function clickStartStop() {
    var $button = $domIds.startStop;
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
    timing.reset();
    stats.reset();
    Samples.reset();
    updateStats();
    $domIds.samples.empty();
  }

  function setupClickHandlers() {
    $domIds.startStop.click(clickStartStop);
    $domIds.reset.click(clickReset);
  }

  function setJqueryIds() {
    var keys = Object.keys(domIds);
    for (var ki = 0; ki < keys.length; ++ki) {
      var key = keys[ki];
      $domIds[key] = $(domIds[key]);
    }
  }

  function main() {
    setJqueryIds();
    net    = new convnetjs.Net();
    wwNet  = new convnetjs.Net({useWorkers: true});
    timing = new Timing();
    stats  = new Stats();
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
