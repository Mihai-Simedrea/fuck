package com.example.fuck

import android.content.Context
import android.graphics.Canvas
import android.graphics.Paint
import android.util.AttributeSet
import android.view.View
import androidx.core.content.ContextCompat
import androidx.core.graphics.ColorUtils


class OverlayTranparent @JvmOverloads constructor(
    context: Context?,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    private val paint: Paint = Paint()

    //position of hole
    var holePosition: OverlayPosition = OverlayPosition(0.0f, 0.0f, 0.0f)
        set(value) {
            field = value
            //redraw
            this.invalidate()
        }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        canvas?.drawRect(0.0f, 0.0f, width.toFloat(), height.toFloat(), paint)
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
    }


}