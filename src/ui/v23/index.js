// v30.23 on-sea UI entry (modern-ui-spec): core (layers, keys, router, overlay) and the hairline HUD first, then the
// world objects in draw order (block handles, section, buoy, sun/moon arc).
import "./core.js?v=32.5";
import "./hud.js?v=32.5";
import "./handles.js?v=31.9";
// After the block handles, so a pointer on the drawn arrow belongs to the wind
// even where that shape crosses another control. The hit uses the same polygon.
import "./arrow.js?v=31.9";
import "./section.js?v=31.2";
import "./ripple.js?v=30.29";
import "./buoy.js?v=31.12";
import "./solar.js?v=30.29";
import "./meters.js?v=33";
