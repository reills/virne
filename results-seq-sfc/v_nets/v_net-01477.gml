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
  id 1477
  arrival_time 32137.07051326432
  lifetime 582.3961753448302
  num_nodes 9
  type "path"
  node [
    id 0
    label "0"
    cpu 18
    gpu 31
    rom 6
  ]
  node [
    id 1
    label "1"
    cpu 34
    gpu 45
    rom 6
  ]
  node [
    id 2
    label "2"
    cpu 38
    gpu 10
    rom 14
  ]
  node [
    id 3
    label "3"
    cpu 14
    gpu 31
    rom 34
  ]
  node [
    id 4
    label "4"
    cpu 4
    gpu 32
    rom 4
  ]
  node [
    id 5
    label "5"
    cpu 31
    gpu 20
    rom 48
  ]
  node [
    id 6
    label "6"
    cpu 18
    gpu 12
    rom 40
  ]
  node [
    id 7
    label "7"
    cpu 23
    gpu 41
    rom 21
  ]
  node [
    id 8
    label "8"
    cpu 0
    gpu 25
    rom 3
  ]
  edge [
    source 0
    target 1
    bw 5
  ]
  edge [
    source 1
    target 2
    bw 17
  ]
  edge [
    source 2
    target 3
    bw 1
  ]
  edge [
    source 3
    target 4
    bw 45
  ]
  edge [
    source 4
    target 5
    bw 40
  ]
  edge [
    source 5
    target 6
    bw 32
  ]
  edge [
    source 6
    target 7
    bw 17
  ]
  edge [
    source 7
    target 8
    bw 11
  ]
]
