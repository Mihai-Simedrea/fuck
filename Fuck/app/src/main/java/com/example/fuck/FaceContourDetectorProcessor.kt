package com.example.fuck;

import android.graphics.Bitmap;
import android.graphics.Color;
import android.graphics.Paint;
import android.util.Log;
import com.google.android.gms.tasks.Task;
import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.common.Triangle;
import com.google.mlkit.vision.face.Face;
import com.google.mlkit.vision.facemesh.FaceMesh;
import com.google.mlkit.vision.facemesh.FaceMeshDetection;
import com.google.mlkit.vision.facemesh.FaceMeshDetector;
import com.google.mlkit.vision.facemesh.FaceMeshDetectorOptions;
import com.google.mlkit.vision.facemesh.FaceMeshPoint;
import java.io.IOException;
import android.graphics.Canvas
import java.io.ByteArrayOutputStream
import java.net.Socket


/**
 * Face Contour Demo.
 */
class FaceContourDetectorProcessor(
    private val faceContourDetectorListener: FaceContourDetectorListener? = null
) : VisionProcessorBase<List<FaceMesh>>() {

    private val detector: FaceMeshDetector = FaceMeshDetection.getClient(
        FaceMeshDetectorOptions.Builder().build()
    );

    private var left = 0F;
    private var right = 0F;
    private var top = 0F;
    private var bottom = 0F;

    private var lastDrawTime: Long = 0L;

    override fun stop() {
        try {
            detector.close();
        } catch (e: IOException) {
            Log.e(TAG, "Exception thrown while trying to close Face Contour Detector: $e");
        }
    }

    override fun detectInImage(image: InputImage): Task<List<FaceMesh>> {
        return detector.process(image);
    }

    override fun onSuccess(
        originalCameraImage: Bitmap?,
        results: List<FaceMesh>,
        graphicOverlay: OverlayView
    ) {
        try {
            val trianglePaint = Paint().apply {
                color = Color.BLUE
                style = Paint.Style.STROKE
                strokeWidth = 1f
            }

            val overlayWidth = graphicOverlay.width.toFloat()
            val overlayHeight = graphicOverlay.height.toFloat()
            val imageWidth = originalCameraImage?.width?.toFloat() ?: overlayWidth
            val imageHeight = originalCameraImage?.height?.toFloat() ?: overlayHeight

            val scaleX = overlayWidth / imageWidth
            val scaleY = overlayHeight / imageHeight

            val canvas = graphicOverlay.lockCanvas()

            for (faceMesh in results) {
                faceMesh.allTriangles.forEach { triangle ->
                    val (point1, point2, point3) = triangle.allPoints.map { it.position }

                    val globalScaler = 1.5f
                    val offset = 250

                    val x1 = overlayWidth - point1.x * scaleX * globalScaler + offset
                    val y1 = point1.y * scaleY
                    val x2 = overlayWidth - point2.x * scaleX * globalScaler + offset
                    val y2 = point2.y * scaleY
                    val x3 = overlayWidth - point3.x * scaleX * globalScaler + offset
                    val y3 = point3.y * scaleY

                    canvas.drawLine(x1, y1, x2, y2, trianglePaint)
                    canvas.drawLine(x2, y2, x3, y3, trianglePaint)
                    canvas.drawLine(x3, y3, x1, y1, trianglePaint)
                }

                val currentTime = System.currentTimeMillis()

                if (currentTime - lastDrawTime >= 2_000) {
                    val boundingBox = faceMesh.boundingBox
                    val croppedBitmap = Bitmap.createBitmap(
                        originalCameraImage ?: continue,
                        boundingBox.left.coerceAtLeast(0),
                        boundingBox.top.coerceAtLeast(0),
                        boundingBox.width().coerceAtMost(originalCameraImage!!.width - boundingBox.left),
                        boundingBox.height().coerceAtMost(originalCameraImage.height - boundingBox.top)
                    )

                    val fixedWidth = 200
                    val fixedHeight = 300

                    val resizedBitmap = Bitmap.createScaledBitmap(
                        croppedBitmap,
                        fixedWidth,
                        fixedHeight,
                        false
                    )

                    val byteArrayOutputStream = ByteArrayOutputStream()
                    resizedBitmap.compress(Bitmap.CompressFormat.PNG, 100, byteArrayOutputStream)
                    val imageBytes = byteArrayOutputStream.toByteArray()

                    Thread {
                        try {
                            val socket = Socket("192.168.175.132", 12345)
                            val outputStream = socket.getOutputStream()
                            outputStream.write(imageBytes)
                            outputStream.flush()
                            socket.close()
                        } catch (e: Exception) {
                            Log.e("SocketError", e.message ?: "Unknown error")
                        }
                    }.start()

                    canvas.drawBitmap(resizedBitmap, 20f, 20f, null)

                    lastDrawTime = currentTime
                }
            }

            graphicOverlay.unlockCanvasAndPost(canvas)
        } catch (e: Exception) {
            Log.e("Exception", e.localizedMessage ?: "Unknown error")
        }
    }

    override fun onFailure(e: Exception) {
        Log.e(TAG, "Face detection failed: ${e.message}");
    }

    private fun translateX(x: Float, overlay: OverlayView): Float {
        return if (overlay.isImageFlipped) {
            overlay.width - (scale(x, overlay) - overlay.postScaleWidthOffset);
        } else {
            x - overlay.postScaleWidthOffset;
        }
    }

    private fun translateY(y: Float, overlay: OverlayView): Float {
        return scale(y, overlay) - overlay.postScaleHeightOffset;
    }

    private fun scale(imagePixel: Float, overlay: OverlayView): Float {
        return imagePixel * overlay.scaleFactor;
    }

    fun isFaceInsideRectangle(faces: List<Face>, graphicOverlay: OverlayView): Boolean {
        return try {
            faces.any { face ->
                val x = translateX(face.boundingBox.centerX().toFloat(), graphicOverlay);
                val y = translateY(face.boundingBox.centerY().toFloat(), graphicOverlay);

                left = x - scale(face.boundingBox.width() / 2.0f, graphicOverlay);
                top = y - scale(face.boundingBox.height() / 2.0f, graphicOverlay);
                right = x + scale(face.boundingBox.width() / 2.0f, graphicOverlay);
                bottom = y + scale(face.boundingBox.height() / 2.0f, graphicOverlay);

                left > graphicOverlay.rectF.left &&
                        top > graphicOverlay.rectF.top &&
                        bottom < graphicOverlay.rectF.bottom &&
                        right < graphicOverlay.rectF.right;
            };
        } catch (e: Exception) {
            false;
        }
    }

    companion object {
        private const val TAG = "FaceContourDetectorProc";
    }

    interface FaceContourDetectorListener {
        fun onCapturedFace(originalCameraImage: Bitmap);
        fun onNoFaceDetected();
    }
}
