// v30.23 on-sea UI entry (modern-ui-spec): core (layers, keys, router, overlay) and the hairline HUD first, then the
// world objects in draw order (block handles, section, buoy, sun/moon arc).
import "./core.js?v=30.27";
import "./hud.js?v=30.27";
import "./handles.js?v=30.27";
// After the block handles, so a pointer on the drawn arrow belongs to the wind
// even where that shape crosses another control. The hit uses the same polygon.
import "./arrow.js?v=30.27";
import "./section.js?v=30.27";
import "./ripple.js?v=30.27";
import "./buoy.js?v=30.27";
import "./solar.js?v=30.27";
