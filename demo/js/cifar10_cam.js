// Author: Peter Jensen

(function () {

  function main() {
    var video = document.querySelector("video");
    navigator.mediaDevices.getUserMedia({video: true})
      .then(function (stream) {
        video.srcObject = stream;
        video.onloadedmetadata = function(e) {
          video.play();
        };
      })
      .catch(function (error) {
      });
  }

  $(main);

})();
