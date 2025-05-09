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
  id 1377
  arrival_time 29198.83066040979
  lifetime 915.4646452004855
  num_nodes 11
  type "path"
  node [
    id 0
    label "0"
    cpu 10
    gpu 45
    rom 50
  ]
  node [
    id 1
    label "1"
    cpu 36
    gpu 5
    rom 39
  ]
  node [
    id 2
    label "2"
    cpu 8
    gpu 26
    rom 23
  ]
  node [
    id 3
    label "3"
    cpu 23
    gpu 11
    rom 39
  ]
  node [
    id 4
    label "4"
    cpu 39
    gpu 37
    rom 4
  ]
  node [
    id 5
    label "5"
    cpu 30
    gpu 45
    rom 24
  ]
  node [
    id 6
    label "6"
    cpu 1
    gpu 5
    rom 32
  ]
  node [
    id 7
    label "7"
    cpu 43
    gpu 40
    rom 43
  ]
  node [
    id 8
    label "8"
    cpu 31
    gpu 5
    rom 24
  ]
  node [
    id 9
    label "9"
    cpu 34
    gpu 43
    rom 41
  ]
  node [
    id 10
    label "10"
    cpu 6
    gpu 2
    rom 36
  ]
  edge [
    source 0
    target 1
    bw 49
  ]
  edge [
    source 1
    target 2
    bw 27
  ]
  edge [
    source 2
    target 3
    bw 1
  ]
  edge [
    source 3
    target 4
    bw 46
  ]
  edge [
    source 4
    target 5
    bw 21
  ]
  edge [
    source 5
    target 6
    bw 2
  ]
  edge [
    source 6
    target 7
    bw 4
  ]
  edge [
    source 7
    target 8
    bw 20
  ]
  edge [
    source 8
    target 9
    bw 32
  ]
  edge [
    source 9
    target 10
    bw 47
  ]
]
