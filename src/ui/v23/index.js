// v30.23 on-sea UI entry (behaviour; the look comes from theme.js). Layers, keys and the
// overlay first, then the world objects, then the right-hand card stack.
import "./core.js";
import "./hud.js";
import "./handles.js";
// After the block handles, so a pointer on the drawn arrow belongs to the wind
// even where that shape crosses another control. The hit uses the same polygon.
import "./arrow.js";
import "./section.js";
import "./ripple.js";
import "./buoy.js";
import "./solar.js";
import "./meters.js";
import "./stack.js";
