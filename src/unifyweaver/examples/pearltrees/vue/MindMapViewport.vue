<template>
  <div
    class="mindmap-viewport"
    :class="{ 'with-grid': showGrid }"
    :style="containerStyle"
    ref="container"
  >
    <svg
      ref="svgRef"
      :width="resolvedWidth"
      :height="resolvedHeight"
      :style="svgStyle"
    >
      <!-- Grid pattern -->
      <defs v-if="showGrid">
        <pattern
          id="grid-pattern"
          :width="gridSize"
          :height="gridSize"
          patternUnits="userSpaceOnUse"
        >
          <path
            :d="`M ${gridSize} 0 L 0 0 0 ${gridSize}`"
            fill="none"
            :stroke="gridColor"
            stroke-width="0.5"
          />
        </pattern>
      </defs>

      <!-- Grid background -->
      <rect
        v-if="showGrid"
        width="100%"
        height="100%"
        fill="url(#grid-pattern)"
      />

      <!-- Content group for transformations -->
      <g ref="contentGroup" class="viewport-content">
        <slot />
      </g>
    </svg>

    <!-- Zoom controls -->
    <div v-if="showControls" class="viewport-controls">
      <button @click="zoomIn" title="Zoom In" class="control-btn">
        <span>+</span>
      </button>
      <button @click="zoomOut" title="Zoom Out" class="control-btn">
        <span>-</span>
      </button>
      <div class="control-separator"></div>
      <button @click="zoomReset" title="Reset (1:1)" class="control-btn">
        <span>1:1</span>
      </button>
      <button @click="fitToContent" title="Fit to Content" class="control-btn">
        <span>[ ]</span>
      </button>
      <div v-if="showZoomIndicator" class="zoom-level">
        {{ zoomPercent }}%
      </div>
    </div>

    <!-- Minimap (optional) -->
    <div v-if="showMinimap" class="minimap">
      <canvas ref="minimapCanvas" :width="minimapSize" :height="minimapSize * 0.75" />
      <div class="minimap-viewport" :style="minimapViewportStyle" />
    </div>
  </div>
</template>

<script setup lang="ts">
/**
 * MindMapViewport.vue
 *
 * Reusable viewport component for mind map visualizations.
 * Generated from UnifyWeaver mindmap_viewport.pl specifications.
 *
 * Features:
 * - Zoom with mouse wheel, pinch, and buttons
 * - Pan with mouse drag
 * - Optional grid background
 * - Optional minimap
 * - Fit-to-content and center functions
 * - Responsive sizing
 *
 * Usage:
 *   <MindMapViewport
 *     :width="800"
 *     :height="600"
 *     :zoom-spec="{ min: 0.1, max: 5, step: 0.1 }"
 *     show-grid
 *     @zoom="handleZoom"
 *     @pan="handlePan"
 *   >
 *     <YourMindMapContent />
 *   </MindMapViewport>
 */

import { ref, computed, onMounted, onUnmounted, watch, type PropType } from 'vue';
import * as d3 from 'd3';

// Types
interface ZoomSpec {
  min?: number;
  max?: number;
  step?: number;
  wheelEnabled?: boolean;
  pinchEnabled?: boolean;
  animationDuration?: number;
}

interface PanSpec {
  enabled?: boolean;
  inertia?: boolean;
  inertiaDecay?: number;
  constrainToBounds?: boolean;
}

interface ViewportState {
  scale: number;
  x: number;
  y: number;
}

interface Props {
  width?: number | string;
  height?: number | string;
  background?: string;
  zoomSpec?: ZoomSpec;
  panSpec?: PanSpec;
  showGrid?: boolean;
  gridSize?: number;
  gridColor?: string;
  showControls?: boolean;
  showZoomIndicator?: boolean;
  showMinimap?: boolean;
  minimapSize?: number;
  initialState?: ViewportState;
}

const props = withDefaults(defineProps<Props>(), {
  width: 800,
  height: 600,
  background: '#ffffff',
  zoomSpec: () => ({ min: 0.1, max: 5, step: 0.1, wheelEnabled: true, pinchEnabled: true, animationDuration: 300 }),
  panSpec: () => ({ enabled: true, inertia: true, inertiaDecay: 0.95, constrainToBounds: false }),
  showGrid: false,
  gridSize: 20,
  gridColor: '#f0f0f0',
  showControls: true,
  showZoomIndicator: true,
  showMinimap: false,
  minimapSize: 150,
  initialState: () => ({ scale: 1, x: 0, y: 0 })
});

const emit = defineEmits<{
  (e: 'zoom', state: ViewportState): void;
  (e: 'pan', state: ViewportState): void;
  (e: 'stateChange', state: ViewportState): void;
}>();

// Refs
const container = ref<HTMLDivElement | null>(null);
const svgRef = ref<SVGSVGElement | null>(null);
const contentGroup = ref<SVGGElement | null>(null);
const minimapCanvas = ref<HTMLCanvasElement | null>(null);

// State
const currentState = ref<ViewportState>({ ...props.initialState });

// D3 references
let zoom: d3.ZoomBehavior<SVGSVGElement, unknown> | null = null;

// Computed
const resolvedWidth = computed(() => {
  if (typeof props.width === 'string') return props.width;
  return props.width;
});

const resolvedHeight = computed(() => {
  if (typeof props.height === 'string') return props.height;
  return props.height;
});

const containerStyle = computed(() => ({
  width: typeof props.width === 'string' ? props.width : `${props.width}px`,
  height: typeof props.height === 'string' ? props.height : `${props.height}px`,
}));

const svgStyle = computed(() => ({
  background: props.background,
}));

const zoomPercent = computed(() => Math.round(currentState.value.scale * 100));

const minimapViewportStyle = computed(() => {
  const scale = props.minimapSize / (typeof props.width === 'number' ? props.width : 800);
  return {
    left: `${-currentState.value.x * scale / currentState.value.scale}px`,
    top: `${-currentState.value.y * scale / currentState.value.scale}px`,
    width: `${props.minimapSize / currentState.value.scale}px`,
    height: `${(props.minimapSize * 0.75) / currentState.value.scale}px`,
  };
});

// Methods
const updateState = (newState: Partial<ViewportState>) => {
  currentState.value = { ...currentState.value, ...newState };
  emit('stateChange', currentState.value);
};

const zoomIn = () => {
  if (!svgRef.value || !zoom) return;
  const duration = props.zoomSpec.animationDuration || 300;
  d3.select(svgRef.value)
    .transition()
    .duration(duration)
    .call(zoom.scaleBy, 1 + (props.zoomSpec.step || 0.1) * 3);
  emit('zoom', currentState.value);
};

const zoomOut = () => {
  if (!svgRef.value || !zoom) return;
  const duration = props.zoomSpec.animationDuration || 300;
  d3.select(svgRef.value)
    .transition()
    .duration(duration)
    .call(zoom.scaleBy, 1 / (1 + (props.zoomSpec.step || 0.1) * 3));
  emit('zoom', currentState.value);
};

const zoomReset = () => {
  if (!svgRef.value || !zoom) return;
  const duration = props.zoomSpec.animationDuration || 300;
  d3.select(svgRef.value)
    .transition()
    .duration(duration)
    .call(zoom.transform, d3.zoomIdentity);
  emit('zoom', currentState.value);
};

const zoomTo = (scale: number) => {
  if (!svgRef.value || !zoom) return;
  const { min = 0.1, max = 5, animationDuration = 300 } = props.zoomSpec;
  const clampedScale = Math.max(min, Math.min(max, scale));
  d3.select(svgRef.value)
    .transition()
    .duration(animationDuration)
    .call(zoom.scaleTo, clampedScale);
};

const fitToContent = () => {
  if (!svgRef.value || !zoom || !contentGroup.value) return;

  const bounds = contentGroup.value.getBBox();
  if (bounds.width === 0 || bounds.height === 0) return;

  const width = typeof props.width === 'number' ? props.width : 800;
  const height = typeof props.height === 'number' ? props.height : 600;
  const { max = 5, animationDuration = 300 } = props.zoomSpec;

  const scale = Math.min(
    (width - 100) / bounds.width,
    (height - 100) / bounds.height,
    max
  );

  const tx = (width - bounds.width * scale) / 2 - bounds.x * scale;
  const ty = (height - bounds.height * scale) / 2 - bounds.y * scale;

  d3.select(svgRef.value)
    .transition()
    .duration(animationDuration)
    .call(zoom.transform, d3.zoomIdentity.translate(tx, ty).scale(scale));
};

const centerOn = (x: number, y: number, scale?: number) => {
  if (!svgRef.value || !zoom) return;

  const width = typeof props.width === 'number' ? props.width : 800;
  const height = typeof props.height === 'number' ? props.height : 600;
  const { animationDuration = 300 } = props.zoomSpec;
  const targetScale = scale ?? currentState.value.scale;

  const tx = width / 2 - x * targetScale;
  const ty = height / 2 - y * targetScale;

  d3.select(svgRef.value)
    .transition()
    .duration(animationDuration)
    .call(zoom.transform, d3.zoomIdentity.translate(tx, ty).scale(targetScale));
};

const getViewportBounds = () => {
  const width = typeof props.width === 'number' ? props.width : 800;
  const height = typeof props.height === 'number' ? props.height : 600;
  const { scale, x, y } = currentState.value;

  return {
    x: -x / scale,
    y: -y / scale,
    width: width / scale,
    height: height / scale,
  };
};

// Initialize viewport
const initViewport = () => {
  if (!svgRef.value) return;

  const svg = d3.select(svgRef.value);
  const { min = 0.1, max = 5, wheelEnabled = true } = props.zoomSpec;

  zoom = d3.zoom<SVGSVGElement, unknown>()
    .scaleExtent([min, max])
    .filter((event) => {
      if (event.type === 'wheel') return wheelEnabled;
      if (event.type === 'mousedown') return event.button === 0;
      return true;
    })
    .on('zoom', (event) => {
      if (contentGroup.value) {
        d3.select(contentGroup.value).attr('transform', event.transform);
      }
      updateState({
        scale: event.transform.k,
        x: event.transform.x,
        y: event.transform.y,
      });
    });

  svg.call(zoom);

  // Apply initial state
  if (props.initialState.scale !== 1 || props.initialState.x !== 0 || props.initialState.y !== 0) {
    const { scale, x, y } = props.initialState;
    svg.call(zoom.transform, d3.zoomIdentity.translate(x, y).scale(scale));
  }
};

// Lifecycle
onMounted(() => {
  initViewport();
});

onUnmounted(() => {
  // Cleanup if needed
});

// Watch for prop changes
watch(() => props.zoomSpec, () => {
  initViewport();
}, { deep: true });

// Expose methods
defineExpose({
  zoomIn,
  zoomOut,
  zoomReset,
  zoomTo,
  fitToContent,
  centerOn,
  getViewportBounds,
  getCurrentState: () => currentState.value,
});
</script>

<style scoped>
.mindmap-viewport {
  position: relative;
  overflow: hidden;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
}

.mindmap-viewport svg {
  display: block;
}

.viewport-controls {
  position: absolute;
  bottom: 16px;
  right: 16px;
  display: flex;
  flex-direction: column;
  gap: 6px;
  z-index: 10;
}

.control-btn {
  width: 36px;
  height: 36px;
  border: none;
  border-radius: 8px;
  background: white;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.12);
  cursor: pointer;
  font-size: 16px;
  font-weight: 500;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.2s ease;
  color: #4a5568;
}

.control-btn:hover {
  background: #f7fafc;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
  transform: translateY(-1px);
}

.control-btn:active {
  transform: translateY(0);
  box-shadow: 0 2px 6px rgba(0, 0, 0, 0.1);
}

.control-separator {
  height: 1px;
  background: #e2e8f0;
  margin: 4px 4px;
}

.zoom-level {
  background: white;
  padding: 6px 10px;
  border-radius: 6px;
  font-size: 12px;
  font-weight: 500;
  text-align: center;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.12);
  color: #4a5568;
  min-width: 48px;
}

.minimap {
  position: absolute;
  bottom: 16px;
  left: 16px;
  background: white;
  border-radius: 8px;
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.15);
  overflow: hidden;
  z-index: 10;
}

.minimap canvas {
  display: block;
}

.minimap-viewport {
  position: absolute;
  border: 2px solid #4a90d9;
  background: rgba(74, 144, 217, 0.1);
  pointer-events: none;
}

/* Grid styling */
.with-grid {
  background-image:
    linear-gradient(to right, var(--grid-color, #f0f0f0) 1px, transparent 1px),
    linear-gradient(to bottom, var(--grid-color, #f0f0f0) 1px, transparent 1px);
  background-size: var(--grid-size, 20px) var(--grid-size, 20px);
}
</style>
