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
  id 1713
  arrival_time 38124.43958430092
  lifetime 21.650283896503886
  num_nodes 9
  type "path"
  node [
    id 0
    label "0"
    cpu 17
    gpu 37
    rom 34
  ]
  node [
    id 1
    label "1"
    cpu 24
    gpu 47
    rom 34
  ]
  node [
    id 2
    label "2"
    cpu 15
    gpu 35
    rom 42
  ]
  node [
    id 3
    label "3"
    cpu 30
    gpu 39
    rom 2
  ]
  node [
    id 4
    label "4"
    cpu 26
    gpu 38
    rom 0
  ]
  node [
    id 5
    label "5"
    cpu 20
    gpu 38
    rom 6
  ]
  node [
    id 6
    label "6"
    cpu 47
    gpu 32
    rom 10
  ]
  node [
    id 7
    label "7"
    cpu 2
    gpu 45
    rom 32
  ]
  node [
    id 8
    label "8"
    cpu 9
    gpu 33
    rom 0
  ]
  edge [
    source 0
    target 1
    bw 38
  ]
  edge [
    source 1
    target 2
    bw 19
  ]
  edge [
    source 2
    target 3
    bw 37
  ]
  edge [
    source 3
    target 4
    bw 4
  ]
  edge [
    source 4
    target 5
    bw 27
  ]
  edge [
    source 5
    target 6
    bw 36
  ]
  edge [
    source 6
    target 7
    bw 29
  ]
  edge [
    source 7
    target 8
    bw 23
  ]
]
