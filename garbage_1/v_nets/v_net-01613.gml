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
  id 1613
  arrival_time 36091.11351740646
  lifetime 455.26428074695247
  num_nodes 9
  type "path"
  node [
    id 0
    label "0"
    cpu 11
    gpu 41
    rom 21
  ]
  node [
    id 1
    label "1"
    cpu 49
    gpu 7
    rom 10
  ]
  node [
    id 2
    label "2"
    cpu 31
    gpu 36
    rom 3
  ]
  node [
    id 3
    label "3"
    cpu 23
    gpu 41
    rom 7
  ]
  node [
    id 4
    label "4"
    cpu 36
    gpu 26
    rom 32
  ]
  node [
    id 5
    label "5"
    cpu 41
    gpu 8
    rom 21
  ]
  node [
    id 6
    label "6"
    cpu 23
    gpu 19
    rom 32
  ]
  node [
    id 7
    label "7"
    cpu 4
    gpu 3
    rom 17
  ]
  node [
    id 8
    label "8"
    cpu 30
    gpu 38
    rom 29
  ]
  edge [
    source 0
    target 1
    bw 35
  ]
  edge [
    source 1
    target 2
    bw 0
  ]
  edge [
    source 2
    target 3
    bw 45
  ]
  edge [
    source 3
    target 4
    bw 24
  ]
  edge [
    source 4
    target 5
    bw 46
  ]
  edge [
    source 5
    target 6
    bw 33
  ]
  edge [
    source 6
    target 7
    bw 40
  ]
  edge [
    source 7
    target 8
    bw 14
  ]
]
