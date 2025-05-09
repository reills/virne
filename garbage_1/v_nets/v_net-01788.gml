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
  id 1788
  arrival_time 39819.143480779516
  lifetime 134.901029630362
  num_nodes 14
  type "path"
  node [
    id 0
    label "0"
    cpu 26
    gpu 40
    rom 46
  ]
  node [
    id 1
    label "1"
    cpu 18
    gpu 39
    rom 36
  ]
  node [
    id 2
    label "2"
    cpu 26
    gpu 6
    rom 17
  ]
  node [
    id 3
    label "3"
    cpu 41
    gpu 27
    rom 33
  ]
  node [
    id 4
    label "4"
    cpu 13
    gpu 35
    rom 36
  ]
  node [
    id 5
    label "5"
    cpu 19
    gpu 16
    rom 8
  ]
  node [
    id 6
    label "6"
    cpu 31
    gpu 21
    rom 47
  ]
  node [
    id 7
    label "7"
    cpu 47
    gpu 47
    rom 50
  ]
  node [
    id 8
    label "8"
    cpu 28
    gpu 5
    rom 49
  ]
  node [
    id 9
    label "9"
    cpu 7
    gpu 38
    rom 15
  ]
  node [
    id 10
    label "10"
    cpu 4
    gpu 47
    rom 38
  ]
  node [
    id 11
    label "11"
    cpu 25
    gpu 39
    rom 26
  ]
  node [
    id 12
    label "12"
    cpu 3
    gpu 45
    rom 35
  ]
  node [
    id 13
    label "13"
    cpu 37
    gpu 46
    rom 10
  ]
  edge [
    source 0
    target 1
    bw 1
  ]
  edge [
    source 1
    target 2
    bw 8
  ]
  edge [
    source 2
    target 3
    bw 28
  ]
  edge [
    source 3
    target 4
    bw 31
  ]
  edge [
    source 4
    target 5
    bw 43
  ]
  edge [
    source 5
    target 6
    bw 15
  ]
  edge [
    source 6
    target 7
    bw 44
  ]
  edge [
    source 7
    target 8
    bw 17
  ]
  edge [
    source 8
    target 9
    bw 45
  ]
  edge [
    source 9
    target 10
    bw 42
  ]
  edge [
    source 10
    target 11
    bw 47
  ]
  edge [
    source 11
    target 12
    bw 33
  ]
  edge [
    source 12
    target 13
    bw 9
  ]
]
