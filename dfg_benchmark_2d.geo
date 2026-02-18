// --------------------
// Parameters
// --------------------
L  = 2.2;  // channel length
H  = 0.41; // channel height
cx = 0.2;  // x-coordinate of the cylinder centre
cy = 0.2;  // y-coordinate of the cylinder centre
r  = 0.05;

h_in   = 0.01;  // target mesh size near the inlet
h_cyl  = 0.002; // target mesh size near the cylinder
h_out  = 0.02;  // target mesh soze near the outlet

// --------------------
// Channel
// --------------------
Point(1) = {0, 0, 0, h_in};
Point(2) = {L, 0, 0, h_out};
Point(3) = {L, H, 0, h_out};
Point(4) = {0, H, 0, h_in};

Line(1) = {1, 2}; // bottom wall
Line(2) = {2, 3}; // outlet
Line(3) = {3, 4}; // top wall
Line(4) = {4, 1}; // inlet

// --------------------
// Cylinder
// --------------------
Point(5) = {cx, cy, 0, h_cyl};

Point(6) = {cx + r, cy, 0, h_cyl};
Point(7) = {cx, cy + r, 0, h_cyl};
Point(8) = {cx - r, cy, 0, h_cyl};
Point(9) = {cx, cy - r, 0, h_cyl};

Circle(5) = {6, 5, 7};
Circle(6) = {7, 5, 8};
Circle(7) = {8, 5, 9};
Circle(8) = {9, 5, 6};

// --------------------
// Surface
// --------------------
Line Loop(1) = {1, 2, 3, 4};
Line Loop(2) = {5, 6, 7, 8};

Plane Surface(1) = {1, 2};

// --------------------
// Physical groups
// --------------------

// Cell tags
Physical Surface(1) = {1};        // fluid domain

// Facet tags
Physical Line(1) = {4};           // inlet
Physical Line(2) = {1, 3};        // top and bottom walls
Physical Line(3) = {5, 6, 7, 8};  // cylinder
Physical Line(4) = {2};           // outlet

// --------------------
// Mesh options
// --------------------
Mesh.Algorithm = 6;
Mesh.MshFileVersion = 4.1;
