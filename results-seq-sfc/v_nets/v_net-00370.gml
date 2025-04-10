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
  id 370
  arrival_time 7010.051845861579
  lifetime 176.84740666980878
  num_nodes 10
  type "path"
  node [
    id 0
    label "0"
    cpu 16
    gpu 48
    rom 34
  ]
  node [
    id 1
    label "1"
    cpu 46
    gpu 49
    rom 40
  ]
  node [
    id 2
    label "2"
    cpu 2
    gpu 36
    rom 32
  ]
  node [
    id 3
    label "3"
    cpu 28
    gpu 27
    rom 17
  ]
  node [
    id 4
    label "4"
    cpu 37
    gpu 50
    rom 8
  ]
  node [
    id 5
    label "5"
    cpu 35
    gpu 36
    rom 27
  ]
  node [
    id 6
    label "6"
    cpu 13
    gpu 38
    rom 13
  ]
  node [
    id 7
    label "7"
    cpu 9
    gpu 13
    rom 33
  ]
  node [
    id 8
    label "8"
    cpu 7
    gpu 43
    rom 41
  ]
  node [
    id 9
    label "9"
    cpu 21
    gpu 3
    rom 44
  ]
  edge [
    source 0
    target 1
    bw 19
  ]
  edge [
    source 1
    target 2
    bw 38
  ]
  edge [
    source 2
    target 3
    bw 47
  ]
  edge [
    source 3
    target 4
    bw 0
  ]
  edge [
    source 4
    target 5
    bw 42
  ]
  edge [
    source 5
    target 6
    bw 37
  ]
  edge [
    source 6
    target 7
    bw 49
  ]
  edge [
    source 7
    target 8
    bw 28
  ]
  edge [
    source 8
    target 9
    bw 45
  ]
]
