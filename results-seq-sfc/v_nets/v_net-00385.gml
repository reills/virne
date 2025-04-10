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
  id 385
  arrival_time 7639.290390164847
  lifetime 617.0405968695119
  num_nodes 8
  type "path"
  node [
    id 0
    label "0"
    cpu 33
    gpu 34
    rom 34
  ]
  node [
    id 1
    label "1"
    cpu 18
    gpu 15
    rom 27
  ]
  node [
    id 2
    label "2"
    cpu 13
    gpu 31
    rom 10
  ]
  node [
    id 3
    label "3"
    cpu 5
    gpu 28
    rom 12
  ]
  node [
    id 4
    label "4"
    cpu 38
    gpu 34
    rom 46
  ]
  node [
    id 5
    label "5"
    cpu 10
    gpu 33
    rom 4
  ]
  node [
    id 6
    label "6"
    cpu 29
    gpu 14
    rom 46
  ]
  node [
    id 7
    label "7"
    cpu 8
    gpu 45
    rom 19
  ]
  edge [
    source 0
    target 1
    bw 8
  ]
  edge [
    source 1
    target 2
    bw 27
  ]
  edge [
    source 2
    target 3
    bw 13
  ]
  edge [
    source 3
    target 4
    bw 22
  ]
  edge [
    source 4
    target 5
    bw 4
  ]
  edge [
    source 5
    target 6
    bw 23
  ]
  edge [
    source 6
    target 7
    bw 50
  ]
]
