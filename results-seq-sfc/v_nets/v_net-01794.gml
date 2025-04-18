graph [
  node_attrs_setting [
    name "cpu"
    distribution "uniform"
    dtype "int"
    generative 1
    low 0
    high 50
    owner "node"
    type "resource"
  ]
  node_attrs_setting [
    name "gpu"
    distribution "uniform"
    dtype "int"
    generative 1
    low 0
    high 50
    owner "node"
    type "resource"
  ]
  node_attrs_setting [
    name "rom"
    distribution "uniform"
    dtype "int"
    generative 1
    low 0
    high 50
    owner "node"
    type "resource"
  ]
  link_attrs_setting "_networkx_list_start"
  link_attrs_setting [
    name "bw"
    distribution "uniform"
    dtype "int"
    generative 1
    low 0
    high 50
    owner "link"
    type "resource"
  ]
  id 1794
  arrival_time 39859.601361098124
  lifetime 1406.4444936111702
  num_nodes 5
  type "path"
  node [
    id 0
    label "0"
    cpu 8
    gpu 0
    rom 16
  ]
  node [
    id 1
    label "1"
    cpu 29
    gpu 27
    rom 20
  ]
  node [
    id 2
    label "2"
    cpu 25
    gpu 12
    rom 26
  ]
  node [
    id 3
    label "3"
    cpu 2
    gpu 1
    rom 17
  ]
  node [
    id 4
    label "4"
    cpu 0
    gpu 16
    rom 21
  ]
  edge [
    source 0
    target 1
    bw 3
  ]
  edge [
    source 1
    target 2
    bw 25
  ]
  edge [
    source 2
    target 3
    bw 36
  ]
  edge [
    source 3
    target 4
    bw 47
  ]
]
