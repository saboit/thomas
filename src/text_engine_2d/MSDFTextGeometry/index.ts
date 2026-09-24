// @ts-ignore-next-line
import createIndices from 'quad-indices'
import { BufferGeometry, Sphere, Box3, BufferAttribute, Matrix4, Vector3 } from 'three'

import TextLayout, { createLayout } from './TextLayout'
import { computeSphere, computeBox } from './utils'
import { generateAttributes } from './vertices'

export interface IInstance {
  text: string
  lineHeight?: number
  letterSpacing?: number
  alignX?: string
  alignY?: string
  width?: number
  yShift?: number
}

export interface IOptions {
  texts: IInstance[]
  flipY?: boolean
  font?: any
  tabSize?: number
  width?: number
  lineHeight?: number
  letterSpacing?: number
  alignX?: string
  alignY?: string
  yShift?: number
}

export interface IGlyph {
  position: [number, number]
  data: any
  index: number
  // Line
  linesTotal: number
  lineIndex: number
  lineLettersTotal: number
  lineLetterIndex: number
  lineWordsTotal: number
  lineWordIndex: number
  // Word
  wordsTotal: number
  wordIndex: number
  // Letter
  lettersTotal: number
  letterIndex: number
}

// TEMP for optimizing allocations
const pointA = new Vector3()
const pointB = new Vector3()
const pointC = new Vector3()
const pointD = new Vector3()

/**
 * The options that decide how one instance looks.
 *
 * Instances with equal values share a layout and its vertices, so the cache is keyed on them. An
 * option that changes the layout but is missing here makes the cache return a stale layout.
 * `alignY` and `yShift` do not move glyphs, but consumers read them from `_layouts[i]._options`.
 */
const LAYOUT_KEYS = ['text', 'width', 'lineHeight', 'letterSpacing', 'alignX', 'alignY', 'yShift', 'tabSize'] as const

/**
 * How many unused instances the cache keeps.
 *
 * A drag removes and adds the same labels many times, so an unused instance is likely to come back.
 */
const CACHE_HEADROOM = 256

type LayoutOptions = Omit<IOptions, 'texts'> & IInstance

interface ICachedInstance {
  layout: TextLayout
  glyphCount: number
  positions: Float32Array
  centers: Float32Array
  uvs: Float32Array
}

function signatureOf(options: LayoutOptions) {
  return JSON.stringify(LAYOUT_KEYS.map((key) => options[key]))
}

function layOut(options: LayoutOptions, texWidth: number, texHeight: number, flipY: boolean): ICachedInstance {
  const layout = createLayout(options.text, options)
  // A glyph with no area draws nothing
  const glyphs = layout.glyphs.filter(({ data }) => data.width * data.height > 0)
  return { layout, glyphCount: glyphs.length, ...generateAttributes(glyphs, texWidth, texHeight, flipY) }
}

export default class MSDFTextGeometry extends BufferGeometry {
  _layouts: TextLayout[]

  originalPositionsArray: Float32Array = new Float32Array()
  positionsArray: Float32Array = new Float32Array()
  centersArray: Float32Array = new Float32Array()
  uvsArray: Float32Array = new Float32Array()
  positionsAttribute: BufferAttribute | null = null

  instancesOffsetsCache: number[] = []
  instancesLengthsCache: number[] = []

  /** Laid-out instances by signature, in least-recently-used order. */
  private cache = new Map<string, ICachedInstance>()
  private cacheFont: any = null
  private cacheFlipY = true
  /** The signatures of the last update. `null` makes the next update run. */
  private signature: string | null = null
  /** `-1` makes the first update build the index and the attributes, also for zero glyphs. */
  private glyphsTotal = -1
  /** The last matrix written per instance. */
  private appliedTransforms: Matrix4[] = []

  constructor(options: IOptions) {
    super()

    this._layouts = []

    this.update(options)
  }

  /**
   * Write the glyph positions of one instance. Returns whether anything was written.
   *
   * Consumers call this for every instance on every frame, and most instances do not move. A write
   * marks the whole shared buffer for upload to the GPU, so an unchanged matrix is skipped.
   */
  setTransform(transform: Matrix4, instance = 0) {
    const instanceOffset = this.instancesOffsetsCache[instance]
    const instanceLength = this.instancesLengthsCache[instance]
    const applied = this.appliedTransforms[instance]

    if (instanceLength === undefined || applied?.equals(transform)) {
      return false
    }

    for (let i = 0; i < instanceLength; i++) {
      const instanceGlyphOffset = instanceOffset * 12 + i * 12

      const originalXA = this.originalPositionsArray[instanceGlyphOffset]
      const originalYA = this.originalPositionsArray[instanceGlyphOffset + 1]
      const originalZA = this.originalPositionsArray[instanceGlyphOffset + 2]
      const originalXB = this.originalPositionsArray[instanceGlyphOffset + 3]
      const originalYB = this.originalPositionsArray[instanceGlyphOffset + 4]
      const originalZB = this.originalPositionsArray[instanceGlyphOffset + 5]
      const originalXC = this.originalPositionsArray[instanceGlyphOffset + 6]
      const originalYC = this.originalPositionsArray[instanceGlyphOffset + 7]
      const originalZC = this.originalPositionsArray[instanceGlyphOffset + 8]
      const originalXD = this.originalPositionsArray[instanceGlyphOffset + 9]
      const originalYD = this.originalPositionsArray[instanceGlyphOffset + 10]
      const originalZD = this.originalPositionsArray[instanceGlyphOffset + 11]

      pointA.set(originalXA, originalYA, originalZA).applyMatrix4(transform)
      pointB.set(originalXB, originalYB, originalZB).applyMatrix4(transform)
      pointC.set(originalXC, originalYC, originalZC).applyMatrix4(transform)
      pointD.set(originalXD, originalYD, originalZD).applyMatrix4(transform)

      this.positionsArray[instanceGlyphOffset] = pointA.x
      this.positionsArray[instanceGlyphOffset + 1] = pointA.y
      this.positionsArray[instanceGlyphOffset + 2] = pointA.z
      this.positionsArray[instanceGlyphOffset + 3] = pointB.x
      this.positionsArray[instanceGlyphOffset + 4] = pointB.y
      this.positionsArray[instanceGlyphOffset + 5] = pointB.z
      this.positionsArray[instanceGlyphOffset + 6] = pointC.x
      this.positionsArray[instanceGlyphOffset + 7] = pointC.y
      this.positionsArray[instanceGlyphOffset + 8] = pointC.z
      this.positionsArray[instanceGlyphOffset + 9] = pointD.x
      this.positionsArray[instanceGlyphOffset + 10] = pointD.y
      this.positionsArray[instanceGlyphOffset + 11] = pointD.z
    }

    // Callers reuse one matrix for all instances, so keep a copy
    this.appliedTransforms[instance] = (applied ?? new Matrix4()).copy(transform)

    if (this.positionsAttribute) {
      this.positionsAttribute.needsUpdate = true
    }
    return true
  }

  update(options: IOptions) {
    const { texts, ...optionsWithoutTexts } = options

    // the desired BMFont data
    const font = options.font

    // get vec2 texcoords
    const flipY = options.flipY !== false

    // A cached layout belongs to the font and the atlas orientation that produced it
    if (font !== this.cacheFont || flipY !== this.cacheFlipY) {
      this.cache.clear()
      this.cacheFont = font
      this.cacheFlipY = flipY
      this.signature = null
    }

    const instancesOptions = texts.map((txt) => ({ ...optionsWithoutTexts, ...txt }))
    const signatures = instancesOptions.map(signatureOf)

    // Consumers call update on every insert and remove, far more often than the content changes
    const signature = signatures.join('\n')
    if (signature === this.signature) {
      return
    }
    this.signature = signature

    // determine texture size from font file
    const texWidth = font.common.scaleW
    const texHeight = font.common.scaleH

    const instances = instancesOptions.map((instanceOptions, i) => {
      const key = signatures[i]
      const instance = this.cache.get(key) ?? layOut(instanceOptions, texWidth, texHeight, flipY)
      // Re-insert, so the instances in use move to the end of the map
      this.cache.delete(key)
      this.cache.set(key, instance)
      return instance
    })

    for (const key of this.cache.keys()) {
      if (this.cache.size <= texts.length + CACHE_HEADROOM) break
      this.cache.delete(key)
    }

    this._layouts = instances.map((instance) => instance.layout)

    // Keep a cache of each text instance offset and length for attributes
    this.instancesOffsetsCache = []
    this.instancesLengthsCache = []
    let offsetSoFar = 0
    instances.forEach(({ glyphCount }, i) => {
      this.instancesLengthsCache[i] = glyphCount
      this.instancesOffsetsCache[i] = offsetSoFar
      offsetSoFar += glyphCount
    })
    const totalGlyphsLength = offsetSoFar

    // Instances move inside the shared buffers, so no applied matrix still holds
    this.appliedTransforms = []

    // The index and the buffer sizes follow from the glyph count. While it stays the same, refill the
    // buffers in place and keep the attributes and the GPU buffers behind them.
    if (totalGlyphsLength !== this.glyphsTotal) {
      this.glyphsTotal = totalGlyphsLength
      this.setIndex(createIndices([], { clockwise: true, type: 'uint16', count: totalGlyphsLength }))

      this.positionsArray = new Float32Array(totalGlyphsLength * 12)
      this.originalPositionsArray = new Float32Array(totalGlyphsLength * 12)
      this.centersArray = new Float32Array(totalGlyphsLength * 8)
      this.uvsArray = new Float32Array(totalGlyphsLength * 8)

      this.positionsAttribute = new BufferAttribute(this.positionsArray, 3)
      this.setAttribute('position', this.positionsAttribute)
      this.setAttribute('center', new BufferAttribute(this.centersArray, 2))
      this.setAttribute('uv', new BufferAttribute(this.uvsArray, 2))
    }

    instances.forEach((instance, i) => {
      const offset = this.instancesOffsetsCache[i]
      this.positionsArray.set(instance.positions, offset * 12)
      this.centersArray.set(instance.centers, offset * 8)
      this.uvsArray.set(instance.uvs, offset * 8)
    })
    this.originalPositionsArray.set(this.positionsArray)

    for (const name of ['position', 'center', 'uv']) {
      this.attributes[name].needsUpdate = true
    }
  }

  override dispose() {
    this.cache.clear()
    this.appliedTransforms = []
    this.signature = null
    super.dispose()
  }

  override computeBoundingSphere() {
    if (this.boundingSphere === null) this.boundingSphere = new Sphere()

    const positions = (this.attributes.position as BufferAttribute).array
    const itemSize = this.attributes.position.itemSize

    if (!positions || !itemSize || positions.length < 2) {
      this.boundingSphere.radius = 0
      this.boundingSphere.center.set(0, 0, 0)
      return
    }

    computeSphere(positions, this.boundingSphere)

    if (isNaN(this.boundingSphere.radius)) {
      // eslint-disable-next-line no-console
      console.error(
        'BufferGeometry.computeBoundingSphere(): Computed radius is NaN. The "position" attribute is likely to have NaN values.'
      )
    }
  }

  override computeBoundingBox() {
    if (this.boundingBox === null) {
      this.boundingBox = new Box3()
    }

    const bbox = this.boundingBox
    const positions = (this.attributes.position as BufferAttribute).array
    const itemSize = this.attributes.position.itemSize

    if (!positions || !itemSize || positions.length < 2) {
      bbox.makeEmpty()
      return
    }

    const box = computeBox(positions, bbox)

    return box
  }
}
