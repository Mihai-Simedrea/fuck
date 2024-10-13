package com.example.fuck

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.PorterDuff
import android.graphics.PorterDuffXfermode
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View
import androidx.core.content.ContextCompat
import androidx.core.graphics.ColorUtils
import com.example.fuck.GraphicOverlay.Graphic

class OverlayPosition(var x: Float, var y: Float, var r: Float)

class OverlayView @JvmOverloads constructor(
    context: Context?,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    private val paint: Paint = Paint()
    var holePaint: Paint = Paint()
    val rectF = RectF()
    private var bitmap: Bitmap? = null
    private var layer: Canvas? = null
    var border: Paint = Paint()
    private var cornerRadius = 30f
    var isImageFlipped = false
    private var needUpdateTransformation = true
    val rectWidth = 300f // Width of the rectangle
    val rectHeight = 200f
    private val lock = Any()
    private val graphics: MutableList<Graphic> = ArrayList()

    // Matrix for transforming from image coordinates to overlay view coordinates.
    private val transformationMatrix = Matrix()
    var imageWidth = 0
        private set
    var imageHeight = 0
        private set

    // The factor of overlay View size to image size. Anything in the image coordinates need to be
    // scaled by this amount to fit with the area of overlay View.
    var scaleFactor = 1.0f

    // The number of horizontal pixels needed to be cropped on each side to fit the image with the
    // area of overlay View after scaling.
    var postScaleWidthOffset = 0f

    // The number of vertical pixels needed to be cropped on each side to fit the image with the
    // area of overlay View after scaling.
    var postScaleHeightOffset = 0f
    private var transparentPaint: Paint? = null
    private var ovalPaint: Paint? = null
    private var centerX = 0
    private  var centerY = 0
    private var ovalWidth = 0
    private  var ovalHeight = 0

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        synchronized(lock) {
            try {
                updateTransformationIfNeeded()
               /* for (graphic in graphics) {
                    graphic.draw(canvas)
                }*/
            } catch (e: Exception) {
            }
        }
        if (bitmap == null) {
            configureBitmap()
        }


        //draw background
        layer?.drawRect(0.0f, 0.0f, width.toFloat(), height.toFloat(), paint)
        //draw hole
      //  layer?.drawCircle((width / 2).toFloat(), (height / 3).toFloat(), 300f, border)

        val centerX = width / 2
        val centerY = height / 3
        val radiusX = width / 2.8
        val radiusY = height / 4

        rectF.set(
            (centerX - radiusX).toFloat(),
            (centerY - radiusY).toFloat(),
            (centerX + radiusX).toFloat(),
            (centerY + radiusY).toFloat()
        )
        layer?.drawRoundRect(rectF, 180F, 180F,border)
        layer?.drawRoundRect(rectF,180F, 180F, holePaint)

      //  layer?.drawCircle((width / 2).toFloat(), (height / 3).toFloat(), 300f, holePaint)

        canvas.drawBitmap(bitmap!!, 0.0f, 0.0f, paint);
    }

    fun clear() {
        synchronized(lock) { graphics.clear() }
        postInvalidate()
    }

    private fun configureBitmap() {
        //create bitmap and layer
        bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        layer = Canvas(bitmap!!)
    }

    fun setImageSourceInfo(imageWidth: Int, imageHeight: Int, isFlipped: Boolean) {
        // Preconditions.checkState(imageWidth > 0, "image width must be positive");
        // Preconditions.checkState(imageHeight > 0, "image height must be positive");
        synchronized(lock) {
            this.imageWidth = imageWidth
            this.imageHeight = imageHeight
            isImageFlipped = isFlipped
            needUpdateTransformation = true
        }
        postInvalidate()
    }

    private fun updateTransformationIfNeeded() {
        try {
            if (!needUpdateTransformation || imageWidth <= 0 || imageHeight <= 0) {
                return
            }
            val viewAspectRatio = width.toFloat() / height
            val imageAspectRatio = imageWidth.toFloat() / imageHeight
            postScaleWidthOffset = 0f
            postScaleHeightOffset = 0f
            if (viewAspectRatio > imageAspectRatio) {
                // The image needs to be vertically cropped to be displayed in this view.
                scaleFactor = width.toFloat() / imageWidth
                postScaleHeightOffset = (width.toFloat() / imageAspectRatio - height) / 2
            } else {
                // The image needs to be horizontally cropped to be displayed in this view.
                scaleFactor = height.toFloat() / imageHeight
                postScaleWidthOffset = (height.toFloat() * imageAspectRatio - width) / 2
            }
            transformationMatrix.reset()
            transformationMatrix.setScale(scaleFactor, scaleFactor)
            transformationMatrix.postTranslate(-postScaleWidthOffset, -postScaleHeightOffset)
            if (isImageFlipped) {
                transformationMatrix.postScale(-1f, 1f, width / 2f, height / 2f)
            }
            needUpdateTransformation = false
        } catch (e: Exception) {
        }
    }

    init {
        //configure background color
        val backgroundAlpha = 0.8
        paint.color = ColorUtils.setAlphaComponent(context?.let {
            ContextCompat.getColor(
                it,
                R.color.overlay
            )
        }!!, (255 * backgroundAlpha).toInt())

        border.color = Color.RED
        border.strokeWidth = 15F
        border.style = Paint.Style.STROKE
        border.isAntiAlias = true
        border.isDither = true

        //configure hole color & mode
        holePaint.color = ContextCompat.getColor(context, android.R.color.transparent)

        holePaint.xfermode = PorterDuffXfermode(PorterDuff.Mode.CLEAR)
    }

    fun add(graphic: Graphic) {
        synchronized(lock) { graphics.add(graphic) }
    }

    /**
     * Removes a graphic from the overlay.
     */
    fun remove(graphic: Graphic) {
        synchronized(lock) { graphics.remove(graphic) }
        postInvalidate()
    }

    abstract class Graphic(private val overlay: OverlayView?) {
        /**
         * Draw the graphic on the supplied canvas. Drawing should use the following methods to convert
         * to view coordinates for the graphics that are drawn:
         *
         *
         *  1. [Graphic.scale] adjusts the size of the supplied value from the image
         * scale to the view scale.
         *  1. [Graphic.translateX] and [Graphic.translateY] adjust the
         * coordinate from the image's coordinate system to the view coordinate system.
         *
         *
         * @param canvas drawing canvas
         */
        abstract fun draw(canvas: Canvas?)

        /**
         * Adjusts the supplied value from the image scale to the view scale.
         */
        fun scale(imagePixel: Float): Float {
            return imagePixel * overlay?.scaleFactor!!
        }

        /**
         * Returns the application context of the app.
         */
        val applicationContext: Context?
            get() = overlay?.context?.applicationContext

        fun isImageFlipped(): Boolean? {
            return overlay?.isImageFlipped
        }

        /**
         * Adjusts the x coordinate from the image's coordinate system to the view coordinate system.
         */
        fun translateX(x: Float): Float {
            return if (overlay?.isImageFlipped == true) {
                overlay.width - (scale(x) - overlay.postScaleWidthOffset)
            } else {
                scale(x) - overlay?.postScaleWidthOffset!!
            }
        }

        /**
         * Adjusts the y coordinate from the image's coordinate system to the view coordinate system.
         */
        fun translateY(y: Float): Float {
            return scale(y) - overlay?.postScaleHeightOffset!!
        }



        /**
         * Returns a [Matrix] for transforming from image coordinates to overlay view coordinates.
         */
        fun getTransformationMatrix(): Matrix? {
            return overlay?.transformationMatrix
        }

        fun postInvalidate() {
            overlay?.postInvalidate()
        }
    }

}