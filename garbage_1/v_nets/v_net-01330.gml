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
  id 1330
  arrival_time 28062.536404044306
  lifetime 333.833488222711
  num_nodes 15
  type "path"
  node [
    id 0
    label "0"
    cpu 4
    gpu 39
    rom 50
  ]
  node [
    id 1
    label "1"
    cpu 26
    gpu 50
    rom 3
  ]
  node [
    id 2
    label "2"
    cpu 5
    gpu 4
    rom 36
  ]
  node [
    id 3
    label "3"
    cpu 27
    gpu 21
    rom 20
  ]
  node [
    id 4
    label "4"
    cpu 50
    gpu 13
    rom 17
  ]
  node [
    id 5
    label "5"
    cpu 0
    gpu 40
    rom 4
  ]
  node [
    id 6
    label "6"
    cpu 25
    gpu 29
    rom 31
  ]
  node [
    id 7
    label "7"
    cpu 18
    gpu 7
    rom 40
  ]
  node [
    id 8
    label "8"
    cpu 24
    gpu 0
    rom 32
  ]
  node [
    id 9
    label "9"
    cpu 31
    gpu 4
    rom 1
  ]
  node [
    id 10
    label "10"
    cpu 41
    gpu 17
    rom 5
  ]
  node [
    id 11
    label "11"
    cpu 3
    gpu 3
    rom 32
  ]
  node [
    id 12
    label "12"
    cpu 44
    gpu 5
    rom 6
  ]
  node [
    id 13
    label "13"
    cpu 25
    gpu 35
    rom 10
  ]
  node [
    id 14
    label "14"
    cpu 12
    gpu 7
    rom 44
  ]
  edge [
    source 0
    target 1
    bw 35
  ]
  edge [
    source 1
    target 2
    bw 30
  ]
  edge [
    source 2
    target 3
    bw 22
  ]
  edge [
    source 3
    target 4
    bw 5
  ]
  edge [
    source 4
    target 5
    bw 41
  ]
  edge [
    source 5
    target 6
    bw 6
  ]
  edge [
    source 6
    target 7
    bw 5
  ]
  edge [
    source 7
    target 8
    bw 7
  ]
  edge [
    source 8
    target 9
    bw 43
  ]
  edge [
    source 9
    target 10
    bw 45
  ]
  edge [
    source 10
    target 11
    bw 15
  ]
  edge [
    source 11
    target 12
    bw 40
  ]
  edge [
    source 12
    target 13
    bw 21
  ]
  edge [
    source 13
    target 14
    bw 26
  ]
]
