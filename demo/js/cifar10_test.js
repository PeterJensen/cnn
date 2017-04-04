// Author: Peter Jensen
(function () {

  const wwQueueCountMax = 4;

  const domIds = {
    startStop:          "#startStop",
    reset:              "#reset",
    useWorkers:         "#useWorkers",
    samples:            "#samples",
    testImageCount:     "#testImageCount",
    testTotalTime:      "#testTotalTime",
    testTimePerImage:   "#testTimePerImage",
    accuracy:           "#accuracy",
    wwTestImageCount:   "#wwTestImageCount",
    wwTestTotalTime:    "#wwTestTotalTime",
    wwTestTimePerImage: "#wwTestTimePerImage",
  };
  var $domIds = {}; // jQuery Ids of referenced DOM elements.  Initialized after page load
  
  // ------------------------
  // BEGIN CIFAR-10 SPECIFIC STUFF
  // ------------------------

  const preTrainedNetFile  = "cifar10_snapshot.json";
  const testBatchImageFile = "cifar10/cifar10_batch_50.png";
  const samplesCount       = 16;
  const testBatch          = 50;
  const numSamplesPerBatch = 1000;
  const imageDimension     = 32;
  const imageChannels      = 3;

  const classes = ['airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'];
  
  // ------------------------
  // END CIFAR-10 SPECIFIC STUFF
  // ------------------------

  // module globals
  var net;
  var wwNet;
  var timer;
  var wwTimers = [];
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
      this.count     = 0;
      this.total     = 0;
      this.startTime = 0;
    }

    Timing.prototype.record = function(fct) {
      var startTime = Date.now();
      fct();
      var stopTime = Date.now();
      this.count++;
      this.total += (stopTime - startTime);
    }
    Timing.prototype.reset = function() {
      this.count     = 0;
      this.total     = 0;
      this.startTime = 0;
    }
    Timing.prototype.start = function() {
      this.startTime = Date.now();
    }
    Timing.prototype.stop = function() {
      this.count++;
      this.total += Date.now() - this.startTime;
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
    $domIds.testImageCount.text(timer.count);
    $domIds.testTotalTime.text(timer.total + "ms");
    if (timer.count > 0) {
      $domIds.testTimePerImage.text(Math.round(timer.total/timer.count) + "ms");
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

    var wwCount = 0;
    var wwTotal = 0;
    for (var i = 0; i < wwQueueCountMax; ++i) {
      wwCount += wwTimers[i].count;
      wwTotal += wwTimers[i].total;
    }
    $domIds.wwTestImageCount.text(wwCount);
    $domIds.wwTestTotalTime.text(wwTotal + "ms");
    if (wwCount > 0) {
      $domIds.wwTestTimePerImage.text(Math.round(wwTotal/wwCount) + "ms");
    }
    else {
      $domIds.wwTestTimePerImage.text("N/A");
    }
  }

  var Processing = (function () {

    function Processing() {}

    // exported static methods

    Processing.start          = start;
    Processing.pause          = pause;
    Processing.useWorkers     = useWorkers;
    Processing.isRunning      = isRunning;
    Processing.isUsingWorkers = isUsingWorkers;

    var stepTimer    = null;
    var usingWorkers = false;
    var running      = false;  // indicate whether the processing should be running
    var queue        = [];

    // local helper functions

    function resume() {
      if (stepTimer !== null) {
        return;
      }
      if (usingWorkers) {
        stepTimer = setInterval(wwProcessNext, 0);
      }
      else {
        stepTimer = setInterval(processNext, 0);
      }
    }

    function suspend() {
      if (stepTimer !== null) {
        clearInterval(stepTimer);
        stepTimer = null;
      }
    }

    function isSuspended() {
      return stepTimer === null;
    }

    function processNext() {
      var sample = Samples.next();
      var result;
      timer.record(function () {
        result = net.forward(sample.vol);
      });
      addSample(sample, result);
      updateStats();
    }

    function wwProcessNext() {
      var sample = Samples.next();
      var wwTimer = wwTimers[queue.length];
      queue.push({sample: sample, timer: wwTimer});
      wwTimer.start();
      wwNet.wwForward(sample.vol, function (result) {
        var entry = queue.shift();
        entry.timer.stop();
        addSample(entry.sample, result);
        updateStats();
        if (running && isSuspended()) {
          resume();
        }
      });
      if (queue.length >= wwQueueCountMax) {
        suspend();
      }
    }

    // exported functions

    function start() {
      running = true;
      resume();
    }

    function pause() {
      running = false;
      suspend();
    }

    function useWorkers(useThem) {
      if (usingWorkers === useThem) {
        return; // do nothing if no change
      }
      usingWorkers = useThem;
      if (running) {
        if (useThem) {
          // Workers weren't used, so start using them
          suspend();
          resume();
        }
        else {
          // Workers were used.  If there's still samples queued up
          // the processing will be resumed when the last one is processed
          suspend();
          if (queue.length === 0) {
            resume();
          }
        }
      }
    }

    function isRunning() {
      return running;
    }

    function isUsingWorkers() {
      return usingWorkers;
    }

    return Processing;

  })();

  // click handlers
  function clickStartStop() {
    var $button = $domIds.startStop;
    if (Processing.isRunning()) {
      log("Pausing");
      $button.text("Start");
      Processing.pause();
    }
    else {
      log("Starting");
      $button.text("Pause");
      Processing.start();
    }
  }

  function clickReset() {
    log("Resetting");
    timer.reset();
    for (var i = 0; i < wwQueueCountMax; ++i) {
      wwTimers[i].reset();
    }
    stats.reset();
    Samples.reset();
    updateStats();
    $domIds.samples.empty();
  }

  function clickUseWorkers() {
    var $button = $domIds.useWorkers;
    if (Processing.isUsingWorkers()) {
      $button.text("Start Workers");
      Processing.useWorkers(false);
    }
    else {
      $button.text("Stop Workers");
      Processing.useWorkers(true);
    }
  }

  function setupClickHandlers() {
    $domIds.startStop.click(clickStartStop);
    $domIds.reset.click(clickReset);
    $domIds.useWorkers.click(clickUseWorkers);
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
    timer  = new Timing();
    for (var i = 0; i < wwQueueCountMax; ++i) {
      wwTimers.push(new Timing());
    }
    stats  = new Stats();
    $.getJSON(preTrainedNetFile, function(json) {
      log(preTrainedNetFile + " loaded");
      net.fromJSON(json);
      wwNet.fromJSON(json);
      Samples.loadBatch(testBatchImageFile, function () {
        log(testBatchImageFile + " loaded");
        setupClickHandlers();
      });
    });
  }

  $(main);

})();
