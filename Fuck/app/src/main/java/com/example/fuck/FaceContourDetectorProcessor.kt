package com.example.fuck;

import android.graphics.Bitmap;
import android.graphics.Color;
import android.graphics.Paint;
import android.util.Log;
import com.google.android.gms.tasks.Task;
import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.facemesh.FaceMesh;
import com.google.mlkit.vision.facemesh.FaceMeshDetector;
import com.google.mlkit.vision.facemesh.FaceMeshDetectorOptions;
import com.google.mlkit.vision.facemesh.FaceMeshDetection;
import android.graphics.Canvas;
import org.tensorflow.lite.Interpreter;
import java.io.ByteArrayOutputStream;
import java.net.Socket;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import org.tensorflow.lite.support.common.FileUtil;
import android.content.Context;
import java.io.IOException;
import java.io.OutputStream

class FaceContourDetectorProcessor(
    private val context: Context,
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
    private var detectedEmotion: String = ""
    private var focusText: String = ""

    private lateinit var socket: Socket
    private lateinit var outputStream: OutputStream

    init {
        Thread {
            try {
                socket = Socket("192.168.60.224", 1235)
                outputStream = socket.getOutputStream()
                Log.d("Socket", "Connected to the server at 192.168.34.132")
                receiveData()
            } catch (e: Exception) {
                Log.e("SocketError", e.message ?: "Unknown error during socket connection")
            }
        }.start()
    }

    private fun receiveData() {
        try {
            val buffer = ByteArray(1024)
            while (true) {
                val bytesRead = socket.getInputStream().read(buffer)
                if (bytesRead > 0) {
                    val receivedData = String(buffer, 0, bytesRead).trim()
                    focusText = if (receivedData == "1") "focused" else "relaxed"
                } else {
                    break
                }
            }
        } catch (e: Exception) {
            Log.e("SocketError", "Error receiving data: ${e.message}")
        }
    }

    override fun stop() {
        try {
            detector.close()
            socket.close()
        } catch (e: IOException) {
            Log.e(TAG, "Exception thrown while trying to close Face Contour Detector: $e")
        }
    }

    override fun detectInImage(image: InputImage): Task<List<FaceMesh>> {
        return detector.process(image);
    }

    private fun preprocessBitmap(bitmap: Bitmap): ByteBuffer {
        val inputBuffer = ByteBuffer.allocateDirect(1 * 48 * 48 * 1 * 4)
        inputBuffer.order(ByteOrder.nativeOrder())

        val intValues = IntArray(48 * 48)
        bitmap.getPixels(intValues, 0, 48, 0, 0, 48, 48)

        for (pixelValue in intValues) {
            val gray = Color.red(pixelValue) * 0.3f + Color.green(pixelValue) * 0.59f + Color.blue(pixelValue) * 0.11f
            inputBuffer.putFloat(gray / 255.0f)
        }

        return inputBuffer
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

                if (currentTime - lastDrawTime >= 500) {
                    val boundingBox = faceMesh.boundingBox
                    val croppedBitmap = Bitmap.createBitmap(
                        originalCameraImage ?: continue,
                        boundingBox.left.coerceAtLeast(0),
                        boundingBox.top.coerceAtLeast(0),
                        boundingBox.width().coerceAtMost(originalCameraImage!!.width - boundingBox.left),
                        boundingBox.height().coerceAtMost(originalCameraImage.height - boundingBox.top)
                    )

                    val resizedBitmap2 = Bitmap.createScaledBitmap(croppedBitmap, 48, 48, false)
                    val inputBuffer = preprocessBitmap(resizedBitmap2)

                    val tfliteModel = FileUtil.loadMappedFile(context, "fer.tflite")
                    val interpreter = Interpreter(tfliteModel)
                    val outputArray = Array(1) { FloatArray(8) }
                    interpreter.run(inputBuffer, outputArray)

                    fun softmax(logits: FloatArray): FloatArray {
                        val expValues = logits.map { Math.exp(it.toDouble()) }
                        val sum = expValues.sum()
                        return expValues.map { (it / sum).toFloat() }.toFloatArray()
                    }

                    val probabilities = softmax(outputArray[0])

                    for (i in probabilities.indices) {
                        Log.e("emotion", "outputArray[0][$i] = ${probabilities[i]}")
                    }

                    if (probabilities[0] < probabilities[1]) {
                        this.detectedEmotion = "Happy"
                    } else {
                        this.detectedEmotion = "Sad"
                    }

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

                    if (this.detectedEmotion == "Sad") {
                        Thread {
                            try {
                                val socket = Socket("192.168.60.147", 12345)
                                val outputStream = socket.getOutputStream()
                                val left = boundingBox.left
                                val top = boundingBox.top
                                val right = boundingBox.right
                                val bottom = boundingBox.bottom

                                val message =
                                    "{\"left\":$left,\"top\":$top,\"right\":$right,\"bottom\":$bottom,\"focusText\":\"$focusText\"}"

                                outputStream.write(message.toByteArray())

                                outputStream.flush()
                                socket.close()
                                Log.d("SocketSuccess", "Message sent: $message")
                            } catch (e: Exception) {
                                Log.e("SocketError", e.message ?: "Unknown error")
                            }
                        }.start()
                    }

                    canvas.drawBitmap(resizedBitmap, 20f, 20f, null)

                    lastDrawTime = currentTime
                }

                val textPaint = Paint().apply {
                    color = Color.WHITE
                    textSize = 72f
                    isAntiAlias = true
                }

                val canvasWidth = canvas.width.toFloat()

                val xPosition = canvasWidth - 250f
                val yPosition = 120f + textPaint.textSize

                canvas.drawText(
                    this.detectedEmotion,
                    xPosition,
                    yPosition,
                    textPaint
                )

                val focusTextPaint = Paint().apply {
                    color = Color.WHITE
                    textSize = 60f
                    isAntiAlias = true
                }

                val focusTextYPosition = yPosition + textPaint.textSize + 40f
                canvas.drawText(
                    this.focusText,
                    xPosition,
                    focusTextYPosition,
                    focusTextPaint
                )
            }

            graphicOverlay.unlockCanvasAndPost(canvas)
        } catch (e: Exception) {
            Log.e("Exception", e.localizedMessage ?: "Unknown error")
        }
    }

    override fun onFailure(e: Exception) {
        Log.e(TAG, "Face detection failed: ${e.message}")
    }

    companion object {
        private const val TAG = "FaceContourDetectorProc"
    }

    interface FaceContourDetectorListener {
        fun onCapturedFace(originalCameraImage: Bitmap)
        fun onNoFaceDetected()
    }
}
