package com.example.fuck

import android.content.res.Resources
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.Rect
import android.util.Log
import com.google.android.gms.tasks.Task
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.common.Triangle
import com.google.mlkit.vision.face.Face
import com.google.mlkit.vision.facemesh.FaceMesh
import com.google.mlkit.vision.facemesh.FaceMeshDetection
import com.google.mlkit.vision.facemesh.FaceMeshDetector
import com.google.mlkit.vision.facemesh.FaceMeshDetectorOptions
import com.google.mlkit.vision.facemesh.FaceMeshPoint
import java.io.IOException
import android.graphics.Paint

/**
 * Face Contour Demo.
 */
class FaceContourDetectorProcessor(
    faceContourDetectorListener: FaceContourDetectorListener? = null,
    isShowDot: Boolean = false
) : VisionProcessorBase<List<FaceMesh>>() {


    private val detector: FaceMeshDetector
    private var mFaceContourDetectorListener: FaceContourDetectorListener? = null
    private var rotationMax = 12
    private var rotationMin = -12
    private var eye = 0.5f
    private var smiling = 0.3f
    private var isMlkit = true
    var left = 0F
    var right = 0F
    var top = 0F
    var bottom = 0F

    init {
//        val options = FaceDetectorOptions.Builder()
//            .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_ACCURATE)
//            .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_ALL)
//            .setContourMode(if (isShowDot) FaceDetectorOptions.CONTOUR_MODE_ALL else FaceDetectorOptions.CONTOUR_MODE_NONE)
//            .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_ALL)
//            .setMinFaceSize(0.5f)
//            .build()

        rotationMax = 12
        rotationMin = -12
        eye = 0.5F
        smiling = 0.0F


        detector = FaceMeshDetection.getClient(
            FaceMeshDetectorOptions.Builder()
                .build()
        )

        mFaceContourDetectorListener = faceContourDetectorListener
    }

    override fun stop() {
        try {
            detector.close()
        } catch (e: IOException) {
            Log.e(TAG, "Exception thrown while trying to close Face Contour Detector: $e")
        }
    }

    override fun detectInImage(image: InputImage): Task<List<FaceMesh>> {
        return detector.process(image)
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

            for (faceMesh in results) {
                val triangles: List<Triangle<FaceMeshPoint>> = faceMesh.allTriangles
                triangles.forEach { triangle ->
                    val connectedPoints = triangle.allPoints

                    val point1 = connectedPoints[0].position
                    val point2 = connectedPoints[1].position
                    val point3 = connectedPoints[2].position

                    val global_scaler = 1.5f
                    val offset = 250

                    val x1 = overlayWidth - point1.x * scaleX * global_scaler + offset
                    val y1 = point1.y * scaleY
                    val x2 = overlayWidth - point2.x * scaleX * global_scaler + offset
                    val y2 = point2.y * scaleY
                    val x3 = overlayWidth - point3.x * scaleX * global_scaler + offset
                    val y3 = point3.y * scaleY

                    val canvas = graphicOverlay.lockCanvas()
                    canvas.drawLine(x1, y1, x2, y2, trianglePaint)
                    canvas.drawLine(x2, y2, x3, y3, trianglePaint)
                    canvas.drawLine(x3, y3, x1, y1, trianglePaint)

                    graphicOverlay.unlockCanvasAndPost(canvas)
                }
            }
        } catch (e: Exception) {
            Log.e("Exception", e.localizedMessage ?: "Unknown error")
        }
    }



    override fun onFailure(e: Exception) {
        try {
            Log.e(TAG, "Face detection failed ${e.message}")
        } catch (e: Exception) {
        }
    }

    private fun extractFace(bmp: Bitmap, x: Int, y: Int, width: Int, height: Int): Bitmap? {
        val originX = if (x + width > bmp.width) (bmp.width - width) else x
        val originY = if (y + height > bmp.height) (bmp.height - height) else y
        return Bitmap.createBitmap(bmp, originX-80, originY-50, width+150, height+150)
    }

    fun translateX(x: Float,overlay: OverlayView): Float {
        return if (overlay.isImageFlipped) {
            overlay.width - (scale(x,overlay) - overlay.postScaleWidthOffset)
        } else {
            x - overlay.postScaleWidthOffset
        }
    }

    fun isFaceInsideRectangle(faces: List<Face>, graphicOverlay: OverlayView):Boolean{
        try {
            faces.forEach { face ->
                val x = translateX(face.boundingBox.centerX().toFloat(),graphicOverlay)
                val y = translateY(face.boundingBox.centerY().toFloat(), graphicOverlay)

                // Draws a bounding box around the face.
                left = x - scale(face.boundingBox.width() / 2.0f, graphicOverlay)
                top = y - scale(face.boundingBox.height() / 2.0f, graphicOverlay)
                right = x + scale(face.boundingBox.width() / 2.0f, graphicOverlay)
                bottom = y + scale(face.boundingBox.height() / 2.0f, graphicOverlay)
            }
            var isFaceInsideRectangle = faces.any { left > graphicOverlay.rectF.left &&
                    top > graphicOverlay.rectF.top &&
                    bottom < graphicOverlay.rectF.bottom &&
                    right < graphicOverlay.rectF.right
            }
            return isFaceInsideRectangle
        }catch (e:Exception){
            return false
        }
    }

    /**
     * Adjusts the y coordinate from the image's coordinate system to the view coordinate system.
     */
    fun translateY(y: Float,overlay:OverlayView): Float {
        return scale(y,overlay) - overlay.postScaleHeightOffset
    }

    fun scale(imagePixel: Float,overlay:OverlayView): Float {
        return imagePixel * overlay.scaleFactor
    }



    companion object {
        private const val TAG = "FaceContourDetectorProc"
    }

    interface FaceContourDetectorListener {
        fun onCapturedFace(originalCameraImage: Bitmap)
        fun onNoFaceDetected()
    }
}